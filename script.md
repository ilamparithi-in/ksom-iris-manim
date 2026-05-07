Imagine you are given a dataset with only one feature. Easy to visualize, right? Now let's take it to two dimensions, two features. What about three features? It's still visualizable. But what if our data has more than three dimensions, ten or a hundred dimensions? It becomes impossible to visualize. The structure exists, but it is hidden. This is exactly the problem that Kohonen set out to solve. He proposed a simple structure, a regular 2D grid. A simple, regular, two dimensional lattice of nodes.This script does not know anything about the data yet. There is no order, structure or relationship to the data. Each node, or neuron of the grid is associated with a weight vector with the same dimensions in the data space. Initially, these weight vectors are randomly arranged in the data space. The grid exists, but its representation is meaningless.

---

Let's take a simpler grid to visualize the learning process. We introduce one data point at a time. We look at the adjacent nodes and ask a simple question: which of these nodes is the closest? By computing the least possible euclidian distance of the input from the weights, we can find the corresponding node or neuron that best represents the input at the moment. This neuron is called the best matching unit or the winner. Once we find the best matching unit, we move it. We take its weight vector, and shift it slightly towards the input. But here is where Kohonen's idea departs from simple clustering. We do not move only the winner. We also move its neighbors. Nodes that are nearby on the grid are also adjusted, though by a smaller amount. Nodes farther away are affected less, or not at all. This influence is controlled by a neighborhood function - a smooth curve that decreases with distance on the grid. So a single input does not just update one neuron. It pulls an entire region of the grid toward itself.

---

If we repeat this process - again and again, for many data points - something remarkable begins to happen. At first, the movements are large. The grid is loose, unstructured, and flexible. Global organization takes place. Gradually, the learning slows down. The adjustments become smaller. The neighborhood shrinks. Now, only nearby nodes move together, refining the structure that has already begun to form. This is the phase of fine tuning. Over time, the weight vectors spread out across the data distribution. The grid, though still flat and regular in its own space, becomes embedded in the data space. It bends, stretches, and aligns itself with the underlying structure of the dataset. And most importantly, it does so while preserving neighborhood relationships.

---

At the end of this process, we are left with something powerful. A two-dimensional map. Each node on the grid corresponds to a vector in the original high-dimensional space. The grid itself becomes a representation of the data distribution - not by projecting points down, but by organizing representative vectors into a structured layout. Nodes that are close together on the grid end up representing similar data points. Nodes that are far apart represent dissimilar ones. The grids represent the data distribution, revealing... its structure, which was hidden from plain sight.

---

Now let us see how we can interpret this map. The weight vectors appear as clusters in the data space. Boundaries emerge from distances, but what defines these boundaries? In the same cluster, the weight vectors are close to each other. Some neighbors, may be close, but different from each other. What if we measure this everywhere? This is what a U-matrix does. It calculates the average distance to the neighbors of a node and structures it in a matrix corresponding to the grid. If the the value in the matrix is high, it means that a boundary was formed.  The regions or clusters formed in the data space by the vectors, and the boundaries, together characterize the self-organized map.

---

Now let's see how we can use the map to identify data points. Let's say we have a new observation. We find the best matching unit and then correspond and map it to the grid. Whichever cluster it is mapped to, that cluster is the cluster, a class of the given data. We can do this for different data points in different areas of the data space. The high-dimensional data has successfully been converted into a 2D representation. The trained self-organizing map is now ready to classify new data points.

---

Even without labels, the self-organizing map already achieves something profound. From disorder... we create... structure. And this is the essence of Kohonen’s idea. Not merely clustering. Not merely dimensionality reduction... But self-organization. A system in which structure is not imposed explicitly, but emerges naturally from local interactions - from simple rules applied repeatedly. A grid that begins as a blank slate becomes a meaningful map. A collection of independent vectors becomes an organized representation.

The end