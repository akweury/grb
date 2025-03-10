
# Gestalt Reasoning Benchmark

This repository provides an overview of the Gestalt Reasoning Benchmark, which evaluates computational models on their ability to recognize and reason using Gestalt principles. The benchmark consists of multiple tasks, each inspired by a fundamental Gestalt principle.

## Benchmark Structure
Each task in the benchmark follows this structure:
- **Categories**: Each Gestalt principle includes multiple script-based categories, each capable of generating multiple tasks.
- **Selected Task for Demonstration**: One task from each category is chosen for illustrative purposes.
- **Three positive examples**: Images that conform to the Gestalt principle.
- **Three negative examples**: Images that do not conform to the principle.
- **Explanation**: A description of the underlying logic behind the positive and negative classifications.

## Tasks

### 1. Proximity
**Principle:** Elements that are close to each other are perceived as a group.

**Category:** `non_overlap_red_triangle`

We show one selected task for demonstration.

#### Task: 001_non_overlap_red_triangle_shape_2_s_all:
<p align="center">
  <span style="color: green; font-weight: bold;">Positive Examples:</span>
  <img src="task_demo/proximity/001_non_overlap_red_triangle_shape_2_s_all/positive/00000.png" width="10%">
  <img src="task_demo/proximity/001_non_overlap_red_triangle_shape_2_s_all/positive/00001.png" width="10%">
  <img src="task_demo/proximity/001_non_overlap_red_triangle_shape_2_s_all/positive/00002.png" width="10%">
  <span style="color: red; font-weight: bold;">Negative Examples:</span>
  <img src="task_demo/proximity/001_non_overlap_red_triangle_shape_2_s_all/negative/00000.png" width="10%">
  <img src="task_demo/proximity/001_non_overlap_red_triangle_shape_2_s_all/negative/00001.png" width="10%">
  <img src="task_demo/proximity/001_non_overlap_red_triangle_shape_2_s_all/negative/00002.png" width="10%">
</p>

**Logic:** In positive examples, there are two groups of proximity clusters, all the clusters contains at least one triangle.

### 2. Similarity

**Principle:** Elements that look similar are perceived as part of the same group.

**Category:** `001_non_overlap_fixed_number_shape_2_s`

**Selected Task for Demonstration:**
<p align="center">
  <span style="color: green; font-weight: bold;">Positive Examples:</span>
  <img src="task_demo/similarity/001/positive/00000.png" width="10%">
  <img src="task_demo/similarity/001/positive/00001.png" width="10%">
  <img src="task_demo/similarity/001/positive/00002.png" width="10%">
  <span style="color: red; font-weight: bold;">Negative Examples:</span>
  <img src="task_demo/similarity/001/negative/00000.png" width="10%">
  <img src="task_demo/similarity/001/negative/00001.png" width="10%">
  <img src="task_demo/similarity/001/negative/00002.png" width="10%">
</p>

**Logic:** In positive examples, two groups of objects are placed in the image, each group of objects have same color. 
The number of objects in two groups are same.

### 3. Closure
**Principle:** The human mind tends to perceive complete figures even when part of the information is missing.

**Category Example:** `001_non_overlap_big_triangle_shape_1_s`

**Selected Task for Demonstration:**
<p align="center">
  <span style="color: green; font-weight: bold;">Positive Examples:</span>
  <img src="task_demo/closure/001_non_overlap_big_triangle_shape_1_s/positive/00000.png" width="10%">
  <img src="task_demo/closure/001_non_overlap_big_triangle_shape_1_s/positive/00001.png" width="10%">
  <img src="task_demo/closure/001_non_overlap_big_triangle_shape_1_s/positive/00002.png" width="10%">
  <span style="color: red; font-weight: bold;">Negative Examples:</span>
  <img src="task_demo/closure/001_non_overlap_big_triangle_shape_1_s/negative/00000.png" width="10%">
  <img src="task_demo/closure/001_non_overlap_big_triangle_shape_1_s/negative/00001.png" width="10%">
  <img src="task_demo/closure/001_non_overlap_big_triangle_shape_1_s/negative/00002.png" width="10%">
</p>

**Logic:** In positive examples, a group of objects form a shape of triangle. 
The shape of an objects can be only circle or square, but not triangle.



### 4. Continuity
**Principle:** Elements arranged in a continuous line or curve are perceived as related.

**Category Example:** `001_non_overlap_big_triangle_shape_1_s`

**Selected Task for Demonstration:**
<p align="center">
  <span style="color: green; font-weight: bold;">Positive Examples:</span>
  <img src="task_demo/closure/001_non_overlap_big_triangle_shape_1_s/positive/00000.png" width="10%">
  <img src="task_demo/closure/001_non_overlap_big_triangle_shape_1_s/positive/00001.png" width="10%">
  <img src="task_demo/closure/001_non_overlap_big_triangle_shape_1_s/positive/00002.png" width="10%">
  <span style="color: red; font-weight: bold;">Negative Examples:</span>
  <img src="task_demo/closure/001_non_overlap_big_triangle_shape_1_s/negative/00000.png" width="10%">
  <img src="task_demo/closure/001_non_overlap_big_triangle_shape_1_s/negative/00001.png" width="10%">
  <img src="task_demo/closure/001_non_overlap_big_triangle_shape_1_s/negative/00002.png" width="10%">
</p>


**Logic:** In positive examples, elements align in a way that creates smooth, continuous patterns. In negative examples, discontinuities disrupt the perceived flow, making elements seem unrelated.


### 4. Continuity
**Principle:** Elements arranged in a continuous line or curve are perceived as related.

**Category Example:** `001_non_overlap_one_split_n_shape_2_s`

**Selected Task for Demonstration:**
<p align="center">
  <span style="color: green; font-weight: bold;">Positive Examples:</span>
  <img src="task_demo/continuity/001_non_overlap_one_split_n_shape_2_s/positive/00000.png" width="10%">
  <img src="task_demo/continuity/001_non_overlap_one_split_n_shape_2_s/positive/00001.png" width="10%">
  <img src="task_demo/continuity/001_non_overlap_one_split_n_shape_2_s/positive/00002.png" width="10%">
  <span style="color: red; font-weight: bold;">Negative Examples:</span>
  <img src="task_demo/continuity/001_non_overlap_one_split_n_shape_2_s/negative/00000.png" width="10%">
  <img src="task_demo/continuity/001_non_overlap_one_split_n_shape_2_s/negative/00001.png" width="10%">
  <img src="task_demo/continuity/001_non_overlap_one_split_n_shape_2_s/negative/00002.png" width="10%">
</p>

**Logic:** In positive examples, a number of objects form a line that splits into two paths at a certain point.
The objects in at least one of the paths have the same shape.



### 5. Symmetry
**Principle:** Symmetrical elements are perceived as belonging together.

**Category Example:** `001_non_overlap_soloar_sys_shape_1`

**Selected Task for Demonstration:**
<p align="center">
  <span style="color: green; font-weight: bold;">Positive Examples:</span>
  <img src="task_demo/symmetry/001_non_overlap_soloar_sys_shape_1/positive/00000.png" width="10%">
  <img src="task_demo/symmetry/001_non_overlap_soloar_sys_shape_1/positive/00001.png" width="10%">
  <img src="task_demo/symmetry/001_non_overlap_soloar_sys_shape_1/positive/00002.png" width="10%">
  <span style="color: red; font-weight: bold;">Negative Examples:</span>
  <img src="task_demo/symmetry/001_non_overlap_soloar_sys_shape_1/negative/00000.png" width="10%">
  <img src="task_demo/symmetry/001_non_overlap_soloar_sys_shape_1/negative/00001.png" width="10%">
  <img src="task_demo/symmetry/001_non_overlap_soloar_sys_shape_1/negative/00002.png" width="10%">
</p>

**Logic:** In positive examples, objects are symmetry align the middle vertical line, 
the left side and right side has same shape. 





## Conclusion
The Gestalt Reasoning Benchmark provides a structured way to evaluate models on visual grouping and perceptual reasoning tasks. Understanding these principles is essential for both human vision and artificial intelligence applications.
