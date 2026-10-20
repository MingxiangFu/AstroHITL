# HITL Feature Lab

This tutorial shows how to label images, train a feature map from those labels, and then train a classifier or a ranker. The [category graph](#3-build-the-category-graph) holds the labels. The [feature map](#5-find-images-of-a-known-type) is trained from that graph. The [classifier](#6-train-a-classifier) and the [ranker](#7-close-the-loop-then-rank) can use either the original features or the mapped features.

The figures use the English labels. The board opens in Chinese. Use the language button in the header to switch.

## Set up

Create a conda environment and install the requirements. Do not install them into the base environment.

```bash
conda create -n hitl-feature-lab python=3.10 -y
conda run -n hitl-feature-lab python -m pip install -r requirements.txt
```

From the repository root, unpack the demo images and start the board:

```bash
mkdir -p subset/images
unzip subset/images.zip -d subset/images
conda run -n hitl-feature-lab python scripts/run_web.py
```

Open `http://127.0.0.1:5002`. The server listens on that address only. The form defaults are the feature file `datasets/fama_base/features.h5` and the image directory `subset/images`. Both paths are resolved from the directory where the server was started. `datasets/ae/features.h5` is an alternate feature table on the same catalog indexes and the same images.

Each project is saved under `web_sessions/`. That folder holds the category graph, the feature map, the classification and ranking models, and the rest of the project. When you start the server again, **Saved projects** lists those projects.

## The board

After a project opens, the page has three columns.

- The left column is the current round. **Suggest** recommends new images. **Current** shows images already in the open category. **Recycle** shows the recycle bin.
- The center column is the [category graph](#3-build-the-category-graph), or a [UMAP](#a1-compare-the-map-with-umap) of the pool. **Graph** and **UMAP** switch that view.
- The right column opens on **Features**, where you [train the map](#5-find-images-of-a-known-type). **Classes** opens [classification and ranking](#6-train-a-classifier).

**Unfrozen** lets you drag category cards to a new position. **Frozen** keeps the cards still, so a drag can drop a card into the recommend well or into a task class. Press `F` to toggle freeze when the cursor is not in a text field.

**Separation** draws each edge by the strength you set for that pair. **Mean distance** draws each edge by the average mapped distance and is the default. Both are described in [Read the mean distance](#a3-read-the-mean-distance).

## 1. Create a project

A project needs a name. Type it in **Project name**. Set the feature file, image directory, pool size, seed, output dimension, round size, and epochs, then click **Create project**.

The name cannot be empty. It cannot repeat a name already in **Saved projects**. It cannot contain `|` or a path separator (`/` or `\`).

**Saved projects** lists every saved project. Choose one and click **Open project**. The opened project includes its [category graph](#3-build-the-category-graph), its [feature map](#5-find-images-of-a-known-type), and its [classification and ranking models](#6-train-a-classifier).

![Create a project: enter a name, then click Create project. Saved projects reopens an earlier one.](tutorial_imgs/1_create_project.png)

## 2. Recommend samples

In **Suggest**, set **Mode** to **Random** and click **Recommend**. The left column fills with unlabeled images from the pool. **Prev** and **Next** move through the round. Under **Random**, the caption is the catalog index. **Certain** and **Uncertain** add a category name and a score, as described in [Find images of a known type](#5-find-images-of-a-known-type).

The modes are:

- **Random** draws images from the pool without using the labels.
- **Diversity** spreads the round across the pool.
- **Unknown** prefers images far from the labeled examples.
- **Certain** prefers images that the categories in scope explain most clearly.
- **Uncertain** prefers images that those categories explain least clearly.

The menu also includes an algorithmic schedule. The first round is random. After that, the schedule uses **Unknown** when two or more categories are in scope, and **Diversity** when fewer than two are in scope.

**Categories**, under the images, chooses which categories are in scope. An empty well means every category at the level you are viewing. Open a category to work on its children, as described in [Add subclasses](#a2-add-subclasses). While the graph is frozen, drag a category card into the well. **All** fills the well with every category at this level. **Clear** empties it.

**×n** is a positive whole number. It starts at 10. It is used when the categories in the well receive no pool image from the nearest-neighbor assignment. In that case the board ranks every pool image by the highest score among those categories, keeps the top `n` times the round size, and runs the selected mode on that group. An image remains in the ranking even when some other category has a higher score for it. When the assignment does give at least one pool image to the categories in the well, the round is chosen from those images. The caption is the category from the well and the score used to rank the row. It is not the category named by a comparison against every labeled category. [Find images of a known type](#5-find-images-of-a-known-type) uses this control with **Certain**.

![Random recommend: choose a mode, then click Recommend.](tutorial_imgs/2_random_recommend.png)

### 2.1 Inspect one image

**Magnifier off · Z** turns the magnifier on. Press `Z` to do the same when the cursor is not in a text field. Hover an image to enlarge it. Press the button or `Z` again to turn it off.

![The magnifier follows the cursor. Toggle it with the button or with Z.](tutorial_imgs/2_1_magnifier.png)

### 2.2 Large figures

**Small** is the default and shows several images in a grid. **Large** shows one image per row. Use it with the magnifier when a thumbnail is not enough.

![Large shows one image per row. Small is the default grid.](tutorial_imgs/2_2_large_fig.png)

## 3. Build the category graph

Drag an image from the left column onto the blank graph. The board asks for a category name and creates a card. Drag another image onto that card to add it to the same category. Each image belongs to one category.

A category name cannot contain `|`. The same name may be used again under a different parent. Two categories with the same parent cannot share a name. While the graph is unfrozen, drag a card to place it. The edges follow the cards. Edge thickness is explained in [Read the mean distance](#a3-read-the-mean-distance).

**Current** lists images in the open category. From a labeled image, **Recycle** moves it into the recycle bin. The image file stays on disk. The card border and the left bar use that category's color.

![Drag an image onto a card to add it. Drag onto empty space to create a category.](tutorial_imgs/3_drag_annotation.png)

## 4. Import a catalog

**Import CSV** reads labels from a file. Choose the file. The dialog asks for the index column, then one or more label columns in level order. **Add label column** adds another level. **Up** and **Down** change that order. Nothing is imported until you click **Import and train**.

[`disk_smooth_demo_input.csv`](disk_smooth_demo_input.csv) is a 10-row sample in this `docs/` folder. Its header is an empty first name, then `index`, then `label`. The `index` column holds catalog ids. The `label` column holds `disk` or `smooth`. In the dialog, set **Index column** to `index` and the label column to `label`.

A label that already exists under the same parent receives the sample. A new name becomes a new category. **Import and train** then trains the [feature map](#5-find-images-of-a-known-type). **Export** writes each placed image with its ancestor names and skips the recycle bin.

![Import CSV opens the column dialog. Import and train merges the labels and trains the map.](tutorial_imgs/4_import_catalog.png)

## 5. Find images of a known type

Train the map from the graph you have built. On the **Features** panel, **Train map** fits that map. **Map width** chooses the hidden width. If you choose a different width, the trained map is cleared and **Train map** fits a new one. **Reset map** removes the trained map. The table under those buttons records, for each round, the mean distance inside a category, the mean distance between categories, and the loss. Clicking an edge is described in [Read the mean distance](#a3-read-the-mean-distance).

After the map is trained, freeze the graph. Drag the category you want into **Categories**. The role of that well is described in [Recommend samples](#2-recommend-samples). Set **Mode** to **Certain** and click **Recommend**. Each caption is the category in scope and the score used to rank that image.

If the nearest-neighbor assignment gives none of the pool images to the categories in the well, **×n** sets how many images are kept before **Certain** runs: `n` times the round size. Change **×n**, then click **Recommend**.

![Frozen graph, disk in the category well, and Certain recommending images for that class.](tutorial_imgs/5_certain_recommend.png)

## 6. Train a classifier

Open **Classes**. Click **New class**, then drag category cards from the graph into that class. Freeze the graph first. While the graph is unfrozen, dragging a card moves it instead of adding it to the class. A task class can hold more than one category. **Rename** and **Delete** edit that class. **Clear** removes every task class. A task class name cannot contain `|`.

**Raw** uses the original features. **Mapped** uses the [feature map](#5-find-images-of-a-known-type) and is available after **Train map** has succeeded. **Model size** is the classifier width for the features you selected. **Train**, **Val**, and **Test** set the split.

**Train classifier** fits the model. The summary reports accuracy and ROC AUC for the split. **Show results** displays a saved result. **Predict** scores the whole pool with the active model. **Show results** and **Predict** share **Result count**. A saved result can be renamed or deleted. The result name cannot contain `|`.

![Freeze the graph, build task classes, train the classifier, then show its predictions.](tutorial_imgs/6_classification.png)

## 7. Close the loop, then rank

Use **Recommend** to see new images, drag them onto the graph, and click **Train map**. Each round adds labels to the same graph. **Train map** continues the same map, unless you changed **Map width** or clicked **Reset map**.

When the categories are ready, open **Classes** and build task classes as in [Train a classifier](#6-train-a-classifier).

**Train ranker** fits a ranker instead of a classifier. Order the task classes, set **Ranking strength** to a positive number, and click **Train ranker**. The summary reports pairwise accuracy. **Predict** and **Show results** also apply to a saved ranking result. Scores are shown highest first, for both classification and ranking.

**Raw** and **Mapped** can be trained on the same task classes, so you can compare the original features with the map.

## Advanced operations

### A.1 Compare the map with UMAP

In the center header, click **UMAP**. The left pane is the original features. The right pane is the mapped features. Gray points are the candidate pool. Colored points are labeled images, one color per category.

Hover a labeled point to see its image and category. The card floats over the plot and does not shift the plots. Click the point to keep the card open until you click another labeled point or empty space. If you have not clicked, moving the cursor off the point hides the card.

The raw plot is calculated once for the current pool. The mapped plot is calculated after the [feature map](#5-find-images-of-a-known-type) is trained, and it is calculated again after each later training of that map.

![Raw and mapped UMAP. Hover a colored point to see the image and its category.](tutorial_imgs/A1_umap.png)

### A.2 Add subclasses

Double-click a category card to open it. The breadcrumb shows the path back to **Root**. **Current** then lists images in that category. **No subclass only** sits with those images, above the recommend-category list. It is on by default and hides images that already have a child category.

Drag an image onto the blank area to create a subclass, or onto a child card to assign it. The drag works as it does in [Build the category graph](#3-build-the-category-graph).

From **Suggest**, a subclass in the **Categories** well is recommended from the full pool. That recommendation needs a trained map and labeled examples of the subclass. From **Current**, **Recommend** uses only images already labeled in the open category.

![Inside a category, double-click opens it. Drag images into subclasses.](tutorial_imgs/A2_subclass.png)

### A.3 Read the mean distance

**Mean distance** is the default edge mode. A thicker edge means the two categories are farther apart after mapping. Click an edge to highlight its row in the table. The row shows the pair and the value. Click a column name to sort, and click it again to reverse the order. **Prev** and **Next** under the table change pages. **Hide edges** closes the table. **Show edges** opens it.

**Separation** uses the same table. The value is then the strength you set for that pair, from 0 to 2, rather than the mapped distance. A thicker edge means the two categories are set to be more separate.

![Click an edge to select its row. Click a column name to sort the table.](tutorial_imgs/A3_mean_distance.png)
