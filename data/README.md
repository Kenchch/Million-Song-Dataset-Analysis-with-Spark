# Data inputs

Do not commit the full Million Song Dataset, course extracts, personal listening histories, cloud credentials, or raw model outputs.

The Spark commands expect these logical shapes:

| Input | Required columns |
| --- | --- |
| Audio attributes CSV | feature name, source type (string, real, numeric, or float) |
| Audio feature CSV | fields matching the attributes CSV, including track_id |
| Genre TSV | track_id, genre |
| Taste Profile TSV | user_id, song_id, play_count |

The exact location and file naming differs between dataset distributions. Adjust paths at the command line and validate the loaded schema before a large run.
