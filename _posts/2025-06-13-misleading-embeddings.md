---
title: "how to lie with embeddings"
description: "visualizations can be misleading"
date: 2025-06-13
permalink: /misleading-embeddings/
layout: post
tag: embeddings, clustering, unsupervised, interaction, sensemaking
custom_css:
  - /assets/css/embedding-viz.css
includes:
  - embedding-viz-scripts.html
---
> Incoherence seems to me preferable to a distorting order. —Barthes

One of the ideas I thought about a lot when studying metaphysics and that continued to find me everywhere since then has been Projectivism. In our perception and understanding of the world, we attribute to it structure that it doesn't actually have. Hume's argument against causation used a form of this diagnosis: we often perceive two events or a series of events in sequence—B always follows A—and imagine that there is some causal relation between A and B, so that A _causes_ B. We never observed the causal relation itself, though, so Hume thinks we have no justification for believing in that relation. All that we observe is correlation. 

Visualizations are useful for understanding data, but they blur the line between what the data actually show and the patterns we project onto data. Every visualization embeds assumptions about the data's structure, and the challenge isn't just validating our assumptions — it's recognizing when we've convinced ourselves we see meaningful patterns that exist only in our interpretation and not in reality. This distinction becomes especially dangerous when the same dataset can be made to tell completely different stories depending on our analytical choices.

## Some Research Behind the Deception

Before we look at experiments, let's understand why these visualizations can be problematic. Several landmark studies have revealed some of our interpretive biases and misunderstandings:

### Visualization Mirages

[Albert et al. (2018)](https://rdc-psychology.org/en/albert_2018) showed that even **randomly sampled** crime incidents produced illusory “hot-spots,” leading participants to re-allocate police resources. Using the VALCRI (Visual Analytics for Sense-making in CRiminal Intelligence analysis) project, the authors asked participants to evaluate whether they would increase police presence in one of two city districts, along with follow-up questions about how the data influenced their decisions and whether they could justify their decisions, given tools that showed spatial and chronological distribution of crime incidents in those two districts. They were given "random condition" data randomly selected from a large set of incidents, "pattern condition" data reflecting real spatial and temporal patterns, and presented with the data in an "interactive condition" where they could interact with tools to inspect incidents from different perspectives as well as a "static condition" where they could not interact with tools.

[McNutt et al. (2020)](https://arxiv.org/pdf/2001.02316) later coined the term **visualization mirages** for silent but significant failures that arise at any stage of the analytic pipeline.

### Cleveland & McGill's Graphical Perception Hierarchy

Cleveland & McGill’s classic experiments rank visual encodings by accuracy: position ≫ length/angle ≫ area ≫ color. Because t-SNE and UMAP re-encode high-dimensional *distance* as **area density and color**, they push viewers toward less reliable perceptual channels—making misreading almost inevitable.

1. **Position along a common scale** (most accurate)
2. **Position along non-aligned scales**
3. **Length, direction, angle**
4. **Area**
5. **Volume, curvature**
6. **Shading, color saturation** (least accurate)

Embedding visualizations face three key challenges: algorithmic sensitivity to parameter choices, method-specific trade-offs between local and global structure preservation, and the gap between what the algorithms optimize for versus what viewers need to interpret. While humans are capable of perceiving patterns and relative distances, different algorithms make different implicit choices about which aspects of high-dimensional structure to prioritize—choices that can dramatically change the story the same data appear to tell.

## A Few Experiments

Let's look at how this happens through a few experiments. I'll limit focus here to t-SNE and UMAP, two popular embedding methods. Both are powerful methods, but need to be treated with some caution, as I'll hope to illustrate below. 

### Experiment 1: Finding Patterns in Pure Randomness

<div id="experiment-1" class="experiment-container"></div>

This experiment demonstrates what researchers call "apophenia"—our tendency to see meaningful patterns in random data. The algorithm parameters don't _just_ reveal structure; they impose some structure that isn't there. 

Research on ensemble perception shows a similar pitfall: observers can summarise large point clouds quickly, but their **subjective confidence often diverges from ground-truth accuracy**.  Two examples are the survey of ensemble coding tasks by [Szafir et al. (2016)](https://jov.arvojournals.org/article.aspx?articleid=2504104) and the “Regression by Eye” experiments by [Correll & Heer (2017)](https://idl.cs.washington.edu/files/2017-RegressionByEye-CHI.pdf).


Wattenberg’s [Distill guide to t-SNE](https://distill.pub/2016/misread-tsne/) (henceforth Wattenberg et al. (2016)) explains that the algorithm expands dense areas and contracts sparse ones—“**cluster sizes … mean nothing**”—and that lowering *perplexity* can manufacture clusters in pure noise.  
UMAP’s own documentation warns that it “**does not completely preserve density**” and “can also create **false tears** in clusters” [(UMAP-learn docs)](https://umap-learn.readthedocs.io/en/latest/clustering.html).  
[Kobak & Berens (2019)](https://www.nature.com/articles/s41467-019-13056-x) provide biological case-studies where such artefacts mislead interpretation, and [Chari & Pachter (2023)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011288) show that, in large single-cell benchmarks, **neighbor-overlap often falls below 0.3**.

### Experiment 2: Hyperparameters matter

<div id="experiment-2" class="experiment-container"></div>

Watch how the same data tells different stories based on parameter choices [(“Those hyper-parameters really matter,” Wattenberg et al., 2016)](https://distill.pub/2016/misread-tsne/#those-hyperparameters-really-matter). Wattenberg et al. (2016) showed that t-SNE is so sensitive to hyperparameters that you can make data look like it has distinct clusters or continuous structure that doesn't exist.

In this experiment, the range is a bit small (perplexity 2-50) and some of the variance only really comes out at higher perplexity values like 100 or so. But you can still see some differences!

### Experiment 3: Confirmation bias / view selection

<div id="experiment-3" class="experiment-container"></div>

Running embeddings until one “looks interesting” (a real thing that happens!) is a visual form of p-hacking. A 2025 CHI study on *confirmation bias* in dashboard “data facts” shows that analysts overwhelmingly choose views that confirm prior beliefs and ignore contradictory ones

## What the Research Tells Us

The research is clear about several critical issues:

### 1. Cluster Assumption

Wattenberg (on t-SNE) and Kobak & Berens both demonstrate that visually separate islands can be artefacts. Users assume that visual clusters represent meaningful groups in the data, but this assumption is often violated by dimensionality reduction algorithms. As noted in the t-SNE literature, visual clusters can appear even in structured data with no clear clustering, making them potentially spurious findings.

### 2. Stability Illusion

Wattenberg also notes that visual “stability” can be forced by parameter tweaking without improving fidelity.

### 3. Narrative Fallacy

Once users see a pattern, they create stories to explain it. Two papers from Cindy Xiong ([2023](https://pubmed.ncbi.nlm.nih.gov/36166548/) and [2019](https://arxiv.org/pdf/1908.00215)) and collaborators offer complementary findings:

1. **Belief-biased estimates (Xiong 2023)** Viewers who *expect* a relationship between two variables over- or under-estimate r-values by ≈0.1.
2. **Causal story-telling from correlation (Xiong 2019)** With the *same* correlational dataset, 33–39 % of study participants who saw a two-bar summary, and ~20 % who saw a scatter plot, wrote explanations that implied causation, despite being reminded that “correlation ≠ causation.” High aggregation (two bars) and grouped encodings produced the strongest causal ratings, while fully disaggregated scatter plots produced the weakest.

## Implications for Practice

Here are a few evidence-based recommendations:

### For Creators of Embeddings

1. **Always show multiple parameter settings** - Single visualizations are misleading by default
2. **Report distance preservation metrics** - Quantify how well distances are preserved
3. **Use stability analysis** - Show how consistent patterns are across runs
4. **Document all preprocessing** - Feature scaling and selection dramatically impact results
5. **Provide interaction** - Let viewers explore the parameter space themselves

### For Consumers of Embeddings

Ask these critical questions:
- What parameters were used? Were they chosen before or after seeing the results?
- How stable are these patterns across different runs?
- What preprocessing was applied to the data?
- How well are distances preserved from the original space?
- What would the visualization look like with different parameters?

## Building Better Practices

The solution isn't to abandon embedding visualizations entirely—they can be useful exploratory tools when used responsibly. The key is to treat them as hypothesis generators, not hypothesis confirmers.

### Validate!

Always validate patterns found in embeddings using other methods:
- Statistical tests in the original high-dimensional space
- Domain expert evaluation
- Predictive modeling to test if clusters are meaningful
- Stability analysis across multiple runs and parameters
- User studies comparing projection methods highlight that some layouts *feel* trustworthy yet score poorly on objective overlap metrics—see the perception-based evaluation by [Etemadpour et al., 2015](https://doi.org/10.1109/TVCG.2014.2330617).

## Toward Honest Visualizations

The research makes one thing clear: our visual system and cognitive biases make us sitting ducks for embedding deceptions. We see patterns where none exist, sometimes create stories to explain randomness, and remain confident in our misinterpretations. This is important, especially if real decisions about research and resource are to be made on the basis of how people interpret these figures. 

On the technical end, we need better tools that communicate uncertainty and parameter sensitivity. Interpretively, we shouldn't think of embedding visualizations as persuasive devices but instead as exploratory tools that we treat with appropriate skepticism.

The ability to create beautiful visualizations is not the same as the ability to reveal truth. Sometimes, the most honest thing we can say about complex data is that it's complex—and no amount of algorithmic magic will change that.

---

## References

- Albert, D. et al. 2018. [Effect of Clustering Illusion during the Interaction with a Visual Analytics Environment](https://rdc-psychology.org/en/albert_2018)
- Chari, T. & Pachter, L. 2023. [The Specious Art of Single-Cell Genomics](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011288)
- Cleveland, W. & McGill, R. 1984. [Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods](http://euclid.psych.yorku.ca/www/psy6135/papers/ClevelandMcGill1984.pdf)
- Correll & Heer, 2017. [Regression by Eye: Estimating Trends in Bivariate Visualizations](https://idl.cs.washington.edu/files/2017-RegressionByEye-CHI.pdf)
- Kobak, D. & Berens, P. 2019. [The Art of Using t-SNE for Single-Cell Transcriptomics](https://www.nature.com/articles/s41467-019-13056-x)
- Li, S. et al. 2025. [Confirmation Bias: The Double-Edged Sword of Data Facts in Visual Data Communication](https://dl.acm.org/doi/10.1145/3706598.3713831)
- McNutt, A., Kindlmann, G., & Correll, M. 2020. ["Surfacing Visualization Mirages"](https://arxiv.org/pdf/2001.02316)
- Etemadpour, R. et al. 2014. [Perception-Based Evaluation of Projection Methods for Multidimensional Data Visualization](https://ieeexplore.ieee.org/document/6832613)
- Szafir et al., 2016. [Four types of ensemble coding in data visualization](https://pubmed.ncbi.nlm.nih.gov/26982369/)
- UMAP-learn documentation, [Using UMAP for Clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html)
- Wattenberg, M., Viégas, F., & Johnson, I. 2016. [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
- Xiong, C. et al., 2019. [Illusion of Causality in Visualized Data](https://arxiv.org/pdf/1908.00215)
- Xiong, C. et al., 2023. ["Seeing What You Believe or Believing What You See? Belief Biases Correlation Estimation"](https://pubmed.ncbi.nlm.nih.gov/36166548/)

<style>
.lead-text {
    font-size: 1.25rem;
    line-height: 1.8;
    color: #555;
    margin-bottom: 2rem;
    font-style: italic;
}

.experiment-container {
    background: #f5f5f5;
    padding: 20px;
    border-radius: 12px;
    margin: 40px auto;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    width: 100%;
    max-width: 100%;
    overflow: hidden;
    box-sizing: border-box;
}

/* Controls styling */
.experiment-controls {
    /* background: white; */
    background: #f5f5f5;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    width: 100%;
}

.control-row {
    margin: 15px 0;
    display: flex;
    align-items: center;
    gap: 15px;
    flex-wrap: wrap;
}

.control-row:last-child {
    margin-bottom: 0;
}

.control-row label {
    min-width: 180px;
    font-weight: 500;
}

.control-row input[type="range"] {
    flex: 1;
    max-width: 300px;
    min-width: 200px;
}

.control-row span {
    min-width: 50px;
    font-weight: 600;
    color: #2196F3;
}

button {
    background: #2196F3;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.3s ease;
    margin-right: 10px;
    margin-bottom: 5px;
}

button:hover {
    background: #1976D2;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

button:disabled {
    background: #ccc;
    cursor: not-allowed;
    transform: none;
}

.viz-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    margin: 20px 0;
    align-items: start;
    width: 100%;
}

.viz-container.single {
    margin: 20px 0;
    width: 100%;
}

.viz-panel {
    background: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    min-height: 400px;
    width: 100%;
    box-sizing: border-box;
    min-width: 0; /* let panels shrink */
}

#experiment-1 .viz-grid {
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    width: 100%;
}

#experiment-1 .viz-panel {
    padding: 15px;
    min-width: 0;
    width: 100%;
    box-sizing: border-box;
}

.viz-panel.large {
    min-height: 400px;
}

.viz-panel h4 {
    margin: 0 0 15px 0;
    font-size: 16px;
    font-weight: 600;
    text-align: center;
    color: #333;
}

.plot-area {
    flex: 1;
    width: 100%;
    min-height: 350px;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    background: #fafafa;
    border-radius: 4px;
    border: 1px solid #e0e0e0;
    overflow: visible;
    padding: 10px;
    box-sizing: border-box;
}

svg {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    display: block;
    max-width: 100%;
    height: auto;
}

.plot-area svg {
    width: 100% !important;
    height: 100% !important;
    max-width: none !important;
    max-height: none !important;
    border: 1px solid #e0e0e0;
    background: #fafafa;
    border-radius: 4px;
    display: block;
}

.point {
    cursor: pointer;
    transition: all 0.2s ease;
}

.point:hover {
    stroke-width: 2px !important;
}

.point.selected {
    stroke: #ff4444 !important;
    stroke-width: 3px !important;
    r: 6 !important;
}

/* Tooltip */
.tooltip {
    position: absolute;
    background: rgba(0,0,0,0.85);
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 12px;
    pointer-events: none;
    z-index: 1000;
    transition: opacity 0.3s;
}

.insight-box {
    background: #e3f2fd;
    padding: 20px;
    border-radius: 8px;
    margin-top: 25px;
    border-left: 4px solid #2196F3;
}

.insight-box.warning {
    background: #fff3cd;
    border-left-color: #ffc107;
}

.insight-box h4 {
    margin-top: 0;
    margin-bottom: 10px;
    color: #1565C0;
}

.insight-box.warning h4 {
    color: #856404;
}

.alert {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    padding: 15px 20px;
    border-radius: 6px;
    margin: 20px 0;
}

.alert strong {
    color: #856404;
}

.parameter-info {
    margin-top: 20px;
    padding: 15px;
    background: #f0f7ff;
    border-radius: 6px;
}

.parameter-info ul {
    margin: 10px 0 0 0;
    padding-left: 20px;
}

.parameter-info li {
    margin: 5px 0;
    color: #555;
}

.status-message {
    margin: 15px 0;
    padding: 10px;
    background: #e8f5e9;
    border-radius: 4px;
    text-align: center;
    font-weight: 500;
    color: #2e7d32;
}

.mini-plots-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    width: 100%;
    padding: 15px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    box-sizing: border-box;
}

.mini-plot {
    background: #f8f9fa;
    padding: 5px;
    border-radius: 6px;
    width: 100%;
    min-width: 100px;
    min-height: 100px;
    aspect-ratio: 1;
    box-sizing: border-box;
    border: 1px solid #e9ecef;
    position: relative;
    overflow: hidden;
}

.mini-plot > div {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.mini-point {
    cursor: default;
}

.mini-plot svg {
    width: 100% !important;
    height: 100% !important;
    border: none;
    background: #f8f9fa;
    border-radius: 3px;
}

.mini-plot .plot-title {
    font-size: 11px;
    font-weight: 600;
    color: #495057;
    margin-bottom: 4px;
    text-align: center;
}

.scale-info {
    margin-top: 20px;
    padding: 15px;
    background: #f9f9f9;
    border-radius: 6px;
    border: 1px solid #e0e0e0;
}

h2 {
    margin-top: 50px;
    margin-bottom: 20px;
    color: #1a1a1a;
}

h3 {
    margin-top: 35px;
    margin-bottom: 15px;
    color: #333;
}

h4 {
    color: #444;
}

p {
    line-height: 1.6;
    margin-bottom: 15px;
}

code {
    background: #f5f5f5;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.9em;
}

.references {
    margin-top: 60px;
    padding-top: 30px;
    border-top: 2px solid #e0e0e0;
}

.references li {
    margin: 10px 0;
    line-height: 1.5;
}

.plot-area > div {
    font-size: 14px;
    color: #64748b;
}

.matrix-container {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.matrix-tooltip {
    z-index: 1000;
}

@media (min-width: 1200px) {
    .experiment-container {
        max-width: 1100px;
        margin-left: auto;
        margin-right: auto;
    }
}

@media (max-width: 900px) {
    #experiment-4 .viz-grid {
        grid-template-columns: 1fr;
    }
    
    #exp4-matrix {
        min-width: 100%;
        max-width: 100%;
    }
}

@media (max-width: 768px) {
    .experiment-container {
        padding: 20px;
        margin: 20px auto;
    }

    #experiment-2 .viz-grid,
    #experiment-4 .viz-grid {
        grid-template-columns: 1fr;
        gap: 15px;
    }

    #experiment-1 .viz-grid {
        grid-template-columns: 1fr 1fr !important;
        gap: 10px;
    }
    .control-row {
        flex-direction: column;
        align-items: flex-start;
        gap: 10px;
    }
    
    .control-row label {
        min-width: 100%;
        margin-bottom: 5px;
    }

    .control-row input[type="range"] {
        width: 100%;
        max-width: 100%;
        min-width: 100%;
    }

    .viz-panel {
        padding: 15px;
        min-height: 350px;
    }

    .plot-area {
        min-height: 250px;
    }

    .mini-plots-grid {
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        padding: 10px;
    }
    
    .mini-plot {
        min-width: 80px;
        min-height: 80px;
    }
}

@media (max-width: 480px) {
    body {
        padding: 10px;
    }
    
    .experiment-container {
        padding: 15px;
    }
    
    button {
        width: 100%;
        margin-right: 0;
        margin-bottom: 10px;
    }

    /* make exp 1 single column */
    #experiment-1 .viz-grid {
        grid-template-columns: 1fr !important;
        gap: 15px;
    }

    .mini-plots-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 6px;
        padding: 8px;
    }
    
    .mini-plot {
        min-width: 70px;
        min-height: 70px;
    }
}

.experiment-container > *:first-child {
    margin-top: 0;
}

.experiment-container > *:last-child {
    margin-bottom: 0;
}

/* give viz-grid for experiment 3 a single column so the
    visible panel (current OR all-results) can take the full width */
#experiment-3 .viz-grid {
    grid-template-columns: 1fr; /* replaces global 1fr 1fr */
}

/* let “all 10 results” panel stretch across
    every grid track if more are ever added dynamically */
#exp3-all-viz {
    grid-column: 1 / -1; /* span entire grid */
}

#exp3-all-viz .mini-plots-grid {
    grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
}

</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script src="/assets/js/embeddings-experiments.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing all experiments...');
    
    if (typeof InteractiveEmbeddingExperiments !== 'undefined' && typeof d3 !== 'undefined') {
        const exp1 = new InteractiveEmbeddingExperiments('experiment-1');
        exp1.initExperiment('random-patterns');
        
        const exp2 = new InteractiveEmbeddingExperiments('experiment-2');
        exp2.initParameterManipulation();
        
        const exp3 = new InteractiveEmbeddingExperiments('experiment-3');
        exp3.initCherryPicking();
    } else {
        console.error('Required libraries not loaded!');
    }
});
</script>