# Source provenance

The following source was integrated into this new application, under the user's
instruction to consolidate their three local projects. The original directories
are not part of the runtime search path and their applications are never launched.

| New code | Origin and integration |
| --- | --- |
| `classical_segmentation/`, `sam2_segmentation/` | Core Python source from graph_catch-main; the new acquisition service owns camera/model lifecycle. |
| `web/shapes.mjs`, editing geometry | Geometry concepts and source from local-annotation-studio. The application, persistence and workflow are newly integrated. |
| `composer_core/geometry.py`, `workbench/pipeline.py` | Dataset Composer's exact-mask geometry and conversion behavior, refactored around the common project snapshot. |
| `store.py`, `server.py`, `desktop.py`, unified UI and orchestration | New integration code. |

Third-party dependencies retain their own licenses. Runtime environments, model
weights, user datasets and generated test artifacts are excluded by `.gitignore`.
Source distribution must retain applicable third-party notices; original project
roots did not contain a project-wide license during this inspection. This document
records provenance and does not replace upstream license texts.
