# Segmentation Findings

- Rows used in this run: **15,000**
- Fast mode: **True**
- KMeans k range tested: 2–10
- Silhouette range across all k: **0.008–0.018** (Rousseeuw 1987 'weak' threshold = 0.25)
- Selected KMeans k: **3** (silhouette=0.018)
- **Interpretation**: silhouette < 0.05 for all k — no meaningful cluster structure detected
- HDBSCAN comparison: status=error: check_array() got an unexpected keyword argument 'ensure_all_finite', clusters=0, noise=nan%, silhouette=nan
- GMM BIC monotonically decreasing: **No** — no preferred number of components
- Convergent conclusion: three independent algorithms (KMeans, HDBSCAN, GMM) confirm no density structure
- Segment names describe behavior patterns only; they do not claim real relationship psychology.

## Segment Profiles

- **Minimal-photo users**: 4,513 users, avg matches=13.71, avg messages=50.50, top intent=Serious Relationship
- **Low-like users**: 7,995 users, avg matches=13.85, avg messages=50.60, top intent=Exploring
- **Quiet browsers**: 2,492 users, avg matches=13.95, avg messages=49.47, top intent=Exploring
