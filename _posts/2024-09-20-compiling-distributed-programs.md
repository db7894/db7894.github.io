---
title:          "compiling distributed ML systems"
description:    "fun"
date:           2024-09-20
permalink:      /compiling-distributed-ml-systems
layout: post
tag: compilers, distributed systems
---

# [Alpa](https://arxiv.org/pdf/2201.12023)
Alpa is a library that automates model-parallel training for larg deep learning models: it automatically generates execution plans that unify data, operator, and pipeline parallelism. Its key idea is viewing parallelisms in two hierarchical levels: _inter-operator_ and _intra-operator_. It designs compilation passes to automatically derive efficient parallel execution plans at each parallelism level. 

## Basics and Definitions
The conventional view of ML parallelization approaches splits them into three categories:
- Data parallelism: partitions the data across devices and trains the model on each partition in parallel. After each worker computes parameter updates on its data split, it needs to synchronize with other workers before performing a weight update. 
- Operator parallelism: partitions the computatin of an operator (e.g. a matmul) along _non-batch_ axes and compute each part of that operator in parallel across multiple devices. This strategy requires communication to fetch input data from other devices. 
- Pipeline parallelism: places different groups of ops from the model graph (stages) on different workers, splits the training batch into microbatches, and pipelines the forward and backward passes across microbatches on distributed workers. 

While approaches such as Megatron-LM manually combine theses three parallelisms, explorations of auto-parallelization relying on this view have limitations. Using an example from the paper: if you're already employing operator parallelism and want to introduce data parallel replicas, you have to introduce a new _set_ of devices and figure out the optimal operator parallelism scheme within those devices. 

The key difference between the types of parallelism presented is their granularity and whether they take operators to be a basic unit.
- data, operator, pipeline parallelism
- intra-operator parallelism: partitions ML operators along one or more tensor axes and dispatches those partitions to distributed devies. This achieves better device utilization, but has larger communication overhead (since it needs to communicate at every split/merge of partitioned operators).
- inter-operator parallelism: slices the model into disjoint stages and pipelines the execution of stages on different sets of devices. With proper slicing the communication overhead can be light, but scheduling constraints cause device idle time (see work like GPipe). 

## Alpa's Strategy
Naturally, then, if you're thinking about mapping parallelism to devices in a compute cluster, it makes sense to map intra-operator parallelism to devices with high communication bandwidth, and inter-operator parallelism to devices with less communication bandwidth. As noted, it works at two key levels:

1. _Intra-op optimization_: Minimize the cost of executing a stage of the computational graph w/r/t/ its intra-operator parallelism plan on a given device mesh (a set of devices with high inter-device bandwidth, e.g. GPUs within a server).
2. _Inter-op optimization_: Minimize inter-op parallelization latency, w/r/t/ slicing the model and device cluster into stages and device meshes and mapping into stage-mesh pairs. This requires knowing the execution cost of each stage-mesh pair reported by the intra-op optimizer. 

It introduces three compiler passes, as shown below:

![Alpa compiler passes and runtime architecture.]({{ site.url }}/assets/images/alpa_figure_3.png)
<p class="pic">Alpa compiler passes and runtime architecture.</p> 


Given Jax IR and a cluster config, the inter-op pass slices the IR into stages, cuts the cluster into device meshes, then assigns stages to device meshes and invokes the intra-op pass on each stage-mesh pair to determine the execution cost of its assignment. It repeatedly queries the intra-op pass and uses DP to minimize inter-op parallel execution latency and achieve the best slicing scheme. You might write it like this, assuming we have certain methods/types:

```
def inter_op_pass(jax_ir: JaxIR, cluster_config: ClusterConfig) -> List[StageMeshPair]:
    num_operators = len(jax_ir.operators)
    num_devices = cluster_config.num_devices
    
    dp = [[float('inf')] * (num_devices + 1) for _ in range(num_operators + 1)]
    dp[0][0] = 0
    backtrack = [[[]] * (num_devices + 1) for _ in range(num_operators + 1)]
    
    # DP
    for i in range(1, num_operators + 1):
        for j in range(1, num_devices + 1):
            for k in range(i):
                stage = jax_ir.operators[k:i]
                for mesh_shape in get_possible_mesh_shapes(j - k):
                    mesh = DeviceMesh.create(mesh_shape)
                    cost = intra_op_pass(stage, mesh, cluster_config)
                    if dp[k][j-len(stage)] + cost < dp[i][j]:
                        dp[i][j] = dp[k][j-len(stage)] + cost
                        backtrack[i][j] = backtrack[k][j-len(stage)] + [StageMeshPair(stage, mesh, cost)]
    
    # return best slicing scheme
    return backtrack[num_operators][num_devices]
```


The intra-op pass solves an ILP to minimize its execution cost. It uses SPMD-style intra-op parallelism to reduce its search space, since SPMD partitions operators evenly across devices  and executes the smae instructions on all devices. 



