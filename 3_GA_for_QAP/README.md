## Description
**Problem**: Quadratic Assignment Problem

**Approach to solving**: Genetic Algorithm

**Parameters**:
* *Number of epochs* = Instance size * 100
* *Population size* = 30
* *Parent selection* = Rank-based
* *Crossover* = Order1
* *Mutation* = Swap
* *Mutation probability* = 0.2
* *Survivor selection* = Rank-based + Elitism
* *Stop criterion*:
  * Epoch limit
  * No improvements plateau = Number of epochs * 0.25

## Results

|Instance|tai20a|tai40a|tai60a|tai80a|tai100a|
|---|---|---|---|---|---|
|**Score**|712948|3261084|7485512|13998870|21789528|

## Evolution history
*Volume* is a sum of scores of all solutions in the population during a given epoch

![](https://github.com/zaktan8/optimization_problems/blob/master/3_GA_for_QAP/images/tai20a.png)
---
![](https://github.com/zaktan8/optimization_problems/blob/master/3_GA_for_QAP/images/tai40a.png)
---
![](https://github.com/zaktan8/optimization_problems/blob/master/3_GA_for_QAP/images/tai60a.png)
---
![](https://github.com/zaktan8/optimization_problems/blob/master/3_GA_for_QAP/images/tai80a.png)
---
![](https://github.com/zaktan8/optimization_problems/blob/master/3_GA_for_QAP/images/tai100a.png)
