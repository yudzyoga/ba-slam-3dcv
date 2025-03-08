To run this code, there are several notebook sequences to follow:
0. (Optional) extract dataset correspondence feature in correspondences_generator.ipynb
1. Preprocess data to label unique points in preprocessing.ipynb
2. Run Bundle Adjustment (same for Stage 1 and Stage 2) in Stage_1_2_BA.ipynb
3. Run SLAM algorithm in Stage_3.ipynb, including the visualization
4. Generate visualization (for BA) in visualization.ipynb
5. Correct the Coordinate system and generate 3D point clouds in postprocess.ipynb 
6. Evaluate BA in eval.ipynb