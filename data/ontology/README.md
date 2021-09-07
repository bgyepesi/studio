<h1>Ontology, Diagnosis Nodes, and Image Counts</h1>

<p>
The "ontology.json" file is common to both the "macroscopic" and "dermoscopic" modalities.
The other files are specific to the modality. In the descriptions below, we focus on the "macroscopic" model,
but the same logic applies to the "dermoscopic" model.
</p>

<h2>ontology.json</h2>
<p>
The current version of the Ontology.
Stores the Triage ID, name, malignancy of each condition, and the hierarchical relationships among skin conditions.
</p>

<h2>macroscopic_diagnosis_nodes.csv</h2>
<p>The diagnosis nodes for the macroscopic model.</p>
<p>
Note that only the Aip IDs should be considered
as the names may be updated in `ontology.json` and not reflected here (although we should try to update the names).
</p>

<h2>macroscopic_image_tags.csv</h2>
<p>The current number of <b>macroscopic</b> images associated with each Aip ID.</p>

<p>You want to scope the image count to only the <b>macroscopic</b> images.</p>
<a href="https://lab.aip.com/datasets/derm/images?tab=tags&tag_query=image-type%3Amacroscopic&tags%5B%5D=&tag_key=AIP">
https://lab.aip.com/datasets/derm/images?tab=tags&tag_query=image-type%3Amacroscopic&tags%5B%5D=&tag_key=AIP</a>

<p>Download the CSV, rename to "macroscopic_image_tags.csv", and run through the notebook example.</p>
