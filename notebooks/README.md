# Original analysis notebooks

These notebooks preserve the code and written analysis from the original distributed Spark project:

1. [`01_processing.ipynb`](01_processing.ipynb) profiles the Million Song data, schemas, genre mappings, audio feature families, and 48.4-million-row Taste Profile input.
2. [`02_audio_similarity.ipynb`](02_audio_similarity.ipynb) joins audio feature families, reduces correlated features, and compares genre classifiers.
3. [`03_song_recommendations.ipynb`](03_song_recommendations.ipynb) filters sparse users and songs, creates a per-user split, trains implicit-feedback ALS, and evaluates top-10 recommendations.

## Security and reproducibility note

The original saved outputs included a Spark configuration table containing expired Azure SAS parameters and a course-cluster username. All outputs and execution counts were therefore removed from the repository copies. Code and Markdown were retained, and the Spark status helper now omits configuration keys matching `secret`, `password`, `token`, `credential`, `sas`, or `account.key`.

The notebooks still refer to the University of Canterbury course environment and Azure Blob paths. They are evidence of the original distributed run, not a promise that the private course infrastructure remains accessible. Use the reusable commands in [`src/msd_pipeline.py`](../src/msd_pipeline.py) for a new environment.
