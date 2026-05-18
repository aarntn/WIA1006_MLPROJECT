# Segmentation Findings

- Rows used in this run: **50,000**
- Fast mode: **False**
- KMeans k range tested: 2–10
- Silhouette range across all k: **0.012–0.019** (Rousseeuw 1987 'weak' threshold = 0.25)
- Selected KMeans k: **3** (silhouette=0.019)
- **Interpretation**: silhouette < 0.05 for all k — no meaningful cluster structure detected
- HDBSCAN comparison: status=ok, clusters=0, noise=100.0%, silhouette=nan
- GMM BIC monotonically decreasing: **No** — no preferred number of components
- Convergent conclusion: three independent algorithms (KMeans, HDBSCAN, GMM) confirm no density structure
- Segment names describe behavior patterns only; they do not claim real relationship psychology.

## Segment Profiles

- **High-like receivers**: 14,851 users, avg matches=13.82, avg messages=50.21, top intent=Networking
- **Low-emoji texters**: 8,276 users, avg matches=13.88, avg messages=49.74, top intent=Serious Relationship
- **Photo-forward users**: 26,873 users, avg matches=13.90, avg messages=50.09, top intent=Casual Dating
