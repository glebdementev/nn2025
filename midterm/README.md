Motivation:

Forest inventory is used to plan maintenance, roads, logging and reforestation efforts. To that end, trees are detected, segmented, classified by species and analyzed to produce timber stand volume estimation.

This project demonstrates point cloud species classification on primitives with noise & jitter. It demonstrates capabilities of PointNet models in tree species classification tasks when using LiDAR data

Tools used:
1. Cursor IDE
2. GPT-5


My Prompts:

1. in midterm folder, please create a file that will generate and classify toy point cloud shapes. use tensor flow js on frontend, no bakcned, no python

it's gotta have an index.html, which shall be styled reasonably and have the following buttons:

1. upload csv
2. generate dataset
3. train
4. evaluate

what it's gotta do is produce TOY shapes of point clouds (not filled)

we have to have the following classes: pyramid, 3d rectangle, cylinder 

nput: point clouds as plain CSV [x,y,z]

Task: classify each cloud as pyramid | box | cylinder

Model: tiny PointNet-style classifier (shared MLP + global max-pool)

have a slider for noise values and slider for jitter

see @Web 

index.html, then other files which should be small, contained, DRY.

make sure to visualize evaluation results

2. 1. make sure that we can view different objects in the viewer, with left and right buttons
2. make sure it generates a 3d object, which we can visualize and look around and rotate
3. make sure we can browse evaluations with their true and assigned metrics

3. 1. allow exporting the generated dataset
2. decompose utils into multiple relevant files
3. boxes, pyramids and cylinders must be of different heights

4. to @index.html 

add explanation why this is important:

simulated tree species classification. a tree crown has a distinct shape (deciduous/conifer) and we may want to train a model to capture this

put it inside a spoiler content

5. see @dataset.js 

make sure that the height varies from 1 of width to 6 of width, but never less. extract this to a slider which assigns left and right see @index.html 

6. see @dataset.js 

in generated dataset, make sure to go classes 1, 2, 3, 1, 2, 3 not 100 of 1, 100 of 2, 100 of 3

7. see @viewer.js 

it has a limit of scale, problem is that we cannot fit the entire thing in viewer. make sure that it autoscales to fit all points

8. see @index.html and @dataset.js 

to add a way to select the classes of generated dataset. also add a new class - ellipsoid, paraboloid, cone, similar to all other classes

9. 1. invert paraboloid on Z axis
2. add a toggle to "fill", which will generate points inside as well

10. see @dataset.js 

when generating a dataset, make a slider on frontend etc. that creates a XY-plane box with random height and width but a slider of min and max share of total floor area of a shape. then the generated shape will be CUT to this shape, Z not considered. the goal is make it more difficult for a model to classify

11. see @index.html 

move training and evaluation above viewer. render evaluation nicer, in a table

render training by just showing a graph of all values and end results, not repeating it for same epoch