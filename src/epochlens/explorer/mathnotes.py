"""LaTeX for the equations EpochLens actually computes. Shared by Streamlit and HTML."""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from io import BytesIO

# Longest first so \\lVert is not eaten by \\lvert.
_MATHTEXT_SUBS: tuple[tuple[str, str], ...] = (
    (r"\lVert", r"\|"),
    (r"\rVert", r"\|"),
    (r"\lvert", r"|"),
    (r"\rvert", r"|"),
    (r"\Bigl", ""),
    (r"\Bigr", ""),
    (r"\bigl", ""),
    (r"\bigr", ""),
)


@dataclass(frozen=True)
class MathNote:
    summary: str
    equations: tuple[str, ...]


NOTES: dict[str, MathNote] = {
    "waveforms": MathNote(
        summary=(
            "Each trial and channel is z-scored from its own baseline window, "
            "then the class mean and standard error are taken over trials. "
            "The plot uses the ranked-channel subset."
        ),
        equations=(
            r"z_{i,c}(t)=\frac{x_{i,c}(t)-\mu_{i,c}^{\mathrm{base}}}{\sigma_{i,c}^{\mathrm{base}}}",
            r"\bar{z}_{c}(t)=\frac{1}{n}\sum_{i=1}^{n} z_{i,c}(t),\quad "
            r"\mathrm{SEM}_{c}(t)=\frac{s_{c}(t)}{\sqrt{n}}",
        ),
    ),
    "spectra": MathNote(
        summary=(
            "Power in the analysis window is the squared magnitude of a real FFT "
            "after a Hann taper. Class curves are trial means of that power. "
            "The y-axis is logarithmic."
        ),
        equations=(
            r"w_t=\frac{1}{2}\bigl(1-\cos\frac{2\pi t}{T-1}\bigr)",
            r"P(f)=\Bigl|\sum_{t=0}^{T-1} w_t\, x_t\, e^{-2\pi i f t/f_s}\Bigr|^2",
        ),
    ),
    "cwt": MathNote(
        summary=(
            "A complex Morlet wavelet is applied in the frequency domain (one FFT "
            "of the data, multiply by each kernel, inverse FFT). Power is "
            r"$|W|^2$, not amplitude. The displayed map is relative to the baseline."
        ),
        equations=(
            r"\sigma_t=\frac{n_{\mathrm{cyc}}}{2\pi f},\quad \sigma_f=\frac{1}{2\pi\sigma_t}",
            r"\Psi_f(\xi)\propto\sqrt{2\sigma_t}\,"
            r"\exp\Bigl(-\frac{1}{2}\bigl((\xi-f)/\sigma_f\bigr)^2\Bigr)",
            r"S(f,t)=|W(f,t)|^2,\quad "
            r"R=\frac{S-\langle S\rangle_{\mathrm{base}}}{\langle S\rangle_{\mathrm{base}}}",
        ),
    ),
    "cwt_energy": MathNote(
        summary="Channel energy is the mean over frequency of the time-variance of relative power in the analysis window.",
        equations=(
            r"E_c=\frac{1}{F}\sum_{f}\mathrm{Var}_{t\in\mathrm{win}}\bigl[R_c(f,t)\bigr]",
        ),
    ),
    "scalp": MathNote(
        summary=(
            "Band power is the mean of the Hann-tapered spectrum inside each band "
            r"$[\theta,\alpha,\beta,\gamma]$. "
            "The scalp field is a thin-plate spline at the 2-D sensor coordinates, "
            "masked to a circle. Each map is scaled on its own."
        ),
        equations=(
            r"B_{c,b}=\mathrm{mean}\bigl\{P_c(f):f\in[f_b^{\mathrm{lo}},f_b^{\mathrm{hi}})\bigr\}",
            r"\phi(\mathbf{r})=\mathrm{TPS}\bigl\{(\mathbf{r}_c,\,B_{c,b})\bigr\}",
        ),
    ),
    "disc": MathNote(
        summary=(
            "At each channel and time sample, class pairs are compared with a "
            "two-sided Mann–Whitney U (README shorthand: Wilcoxon). "
            r"$U$ is converted to a normal $z$ under the null; the map shows $|z|$. "
            "Ranking averages $|z|$ over pairs."
        ),
        equations=(
            r"\mu_U=\frac{n_1 n_2}{2},\quad "
            r"\sigma_U=\sqrt{\frac{n_1 n_2(n_1+n_2+1)}{12}}",
            r"z=\frac{U-\mu_U}{\sigma_U},\quad "
            r"s_c=\mathrm{mean}_{\mathrm{pairs},\,t}\lvert z_{c,t}\rvert",
        ),
    ),
    "ranking": MathNote(
        summary=(
            "Channels are sorted by that discriminability score. Dead, single-trial-dominated, "
            "or MAD-outlier sensors (log RMS) are excluded. Waveforms / CWT / spectra use "
            "the top-$k$ subset; MDS and the chance check do not."
        ),
        equations=(
            r"\mathrm{MAD}=1.4826\cdot\mathrm{median}_c\lvert \log_{10}\mathrm{RMS}_c-m\rvert",
            r"\mathrm{flag}_c=\mathbf{1}\bigl[\lvert\log_{10}\mathrm{RMS}_c-m\rvert "
            r"> 8\cdot\max(\mathrm{MAD},0.15)\bigr]",
        ),
    ),
    "mds": MathNote(
        summary=(
            "Each trial is a trace-normalized covariance of the analysis window "
            "(all channels), with a small ridge. Distances are log-Euclidean: "
            "Frobenius distance of matrix logarithms. Points are classical MDS. "
            "Session whitening removes the per-session log-Euclidean mean."
        ),
        equations=(
            r"C_i=\frac{X_i X_i^\top}{T\,\mathrm{tr}(X_i X_i^\top/T)}+\lambda I",
            r"d(A,B)=\lVert\log A-\log B\rVert_F",
            r"B=-\frac{1}{2} H\,D^{\circ 2}\,H,\quad "
            r"Y=V_{1:2}\,\mathrm{diag}(\sqrt{\lambda_{1:2}})",
            r"C\leftarrow R^{-1/2} C R^{-1/2}",
        ),
    ),
    "chance": MathNote(
        summary=(
            "The same covariances are mapped to the upper triangle of "
            r"$\log C$, standardized, and classified with shrinkage LDA under "
            "stratified cross-validation. Chance is $1/K$; majority is the "
            "largest class fraction. This is a sanity check, not a BCI."
        ),
        equations=(
            r"\mathbf{v}_i=\mathrm{vech}(\log C_i)",
            r"\mathrm{chance}=1/K,\quad "
            r"\mathrm{majority}=\max_k n_k/N",
        ),
    ),
}


def mathtext_safe(eq: str) -> str:
    """Map display LaTeX to matplotlib mathtext."""
    out = eq
    for src, dst in _MATHTEXT_SUBS:
        out = out.replace(src, dst)
    return out


def equation_png(eq: str) -> str:
    """Render one display equation to a PNG data-URL payload (no CDN)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.6, 0.52))
    fig.patch.set_facecolor("#f7f1e6")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        f"${mathtext_safe(eq)}$",
        ha="center",
        va="center",
        fontsize=12,
        color="#1C1917",
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.06)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def html_block(key: str) -> str:
    note = NOTES[key]
    eqs = "".join(
        f'<p class="eq"><img alt="" src="data:image/png;base64,{equation_png(eq)}"/></p>'
        for eq in note.equations
    )
    summary = html.escape(note.summary.replace("$", ""))
    return f"<div class=\"math\"><p>{summary}</p>{eqs}</div>"


def show_math(st, key: str) -> None:
    note = NOTES[key]
    with st.expander("Math"):
        st.markdown(note.summary)
        for eq in note.equations:
            st.latex(eq)
