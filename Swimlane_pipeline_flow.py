#!/usr/bin/env python3
# Draws a swimlane diagram for: Scraper → ML → LLM → Human Reviewer → Truth DB
# Requirements: matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def lane(ax, y, h, label):
    ax.plot([0, 36], [y, y], linewidth=1)
    ax.text(0.5, y + h/2, label, va='center', ha='left', fontsize=13, fontweight='bold')

def box(ax, x, y, w, h, title, items=()):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.1",
                          linewidth=1.4, facecolor="none")
    ax.add_patch(rect)
    text = title + ("\n" + "\n".join(f"• {it}" for it in items) if items else "")
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, wrap=True)

def harrow(ax, x1, y, x2, label=None):
    a = FancyArrowPatch((x1, y), (x2, y), arrowstyle='->', mutation_scale=12, linewidth=1.3)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, y+0.25, label, ha='center', va='bottom', fontsize=10)

def varrow(ax, x, y1, y2, label=None):
    a = FancyArrowPatch((x, y1), (x, y2), arrowstyle='->', mutation_scale=12, linewidth=1.3)
    ax.add_patch(a)
    if label:
        ax.text(x+0.25, (y1+y2)/2, label, va='center', ha='left', fontsize=10)

def draw(output_png="swimlane_worker_pipeline_flow.png", output_pdf="swimlane_worker_pipeline_flow.pdf"):
    plt.figure(figsize=(22, 12), dpi=150)
    ax = plt.gca()
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # Lane positions
    H = 3
    Y = {
        "Scraper / Parser": 13,
        "ML Models": 10,
        "LLM Verifier": 7,
        "Human Reviewer": 4,
        "Truth DB / Exports": 1
    }
    for name, y in Y.items():
        lane(ax, y, H, name)

    # Column centers and sizes
    col = {1:5, 2:11, 3:17, 4:23, 5:29}
    W, BH = 5.5, 2.4

    # Scraper / Parser
    box(ax, col[1]-W/2, Y["Scraper / Parser"]+0.4, W, BH, "extractors.py",
        ["download_pdf(url)", "camelot/pdfplumber", "extract_tables(pdf)"])
    box(ax, col[2]-W/2, Y["Scraper / Parser"]+0.4, W, BH, "pipeline.py",
        ["coerce_headers()", "header_map()", "dataframe_to_payloads"])
    harrow(ax, col[1]+W/2, Y["Scraper / Parser"]+1.5, col[2]-W/2, "PDFs → DataFrames")

    # ML Models
    box(ax, col[2]-W/2, Y["ML Models"]+0.4, W, BH, "parsers.py",
        ["parse_cell()", "acres_to_sq_ft()", "extract_depth()"])
    box(ax, col[3]-W/2, Y["ML Models"]+0.4, W, BH, "mapping.py",
        ["header_map()", "load_profile()", "state/muni profiles"])
    box(ax, col[4]-W/2, Y["ML Models"]+0.4, W, BH, "validators.py",
        ["compute_confidence()", "REQUIRED fields", "unit canonicalization"])

    # Connect scraper to ML
    varrow(ax, col[2], Y["Scraper / Parser"]+0.4, Y["ML Models"]+BH+0.2, "zone payloads")
    harrow(ax, col[2]+W/2, Y["ML Models"]+1.5, col[3]-W/2, "parsed cells")
    harrow(ax, col[3]+W/2, Y["ML Models"]+1.5, col[4]-W/2, "validated data")

    # LLM Verifier
    box(ax, col[4]-W/2, Y["LLM Verifier"]+0.4, W, BH, "main.py",
        ["process_job()", "zone_groups{}", "consolidate payloads"])
    box(ax, col[5]-W/2, Y["LLM Verifier"]+0.4, W, BH, "verifier.py",
        ["confidence check", "AUTO_INGEST flag", "CONF_THRESH=0.90"])

    # Connect ML to LLM
    varrow(ax, col[4], Y["ML Models"]+0.4, Y["LLM Verifier"]+BH+0.2, "zone data")
    harrow(ax, col[4]+W/2, Y["LLM Verifier"]+1.5, col[5]-W/2, "for review")

    # Human Reviewer
    box(ax, col[3]-W/2, Y["Human Reviewer"]+0.4, W, BH, "Admin Dashboard",
        ["View raw_extractions", "Manual corrections"])
    box(ax, col[5]-W/2, Y["Human Reviewer"]+0.4, W, BH, "Review UI",
        ["Accept/Reject zones", "Edit standards", "Override confidence"])

    # Connect LLM to Human Reviewer
    varrow(ax, col[5], Y["LLM Verifier"]+0.4, Y["Human Reviewer"]+BH+0.2, "low conf")
    harrow(ax, col[5]-W/2-0.2, Y["Human Reviewer"]+1.5, col[3]+W/2+0.2, "review queue")

    # Truth DB / Exports
    box(ax, col[4]-W/2, Y["Truth DB / Exports"]+0.4, W, BH, "supa.py",
        ["save_raw()", "call_admin_ingest()", "upsert zones/standards"])
    box(ax, col[3]-W/2, Y["Truth DB / Exports"]+0.4, W, BH, "Supabase DB",
        ["zones table", "zone_standards", "ingestion_jobs"])
    box(ax, col[2]-W/2, Y["Truth DB / Exports"]+0.4, W, BH, "Frontend App",
        ["Search interface", "Zone details", "Standards view"])

    # Connections from LLM / Reviewer to DB
    varrow(ax, col[4]+0.5, Y["LLM Verifier"]+0.4, Y["Truth DB / Exports"]+BH+0.2, "auto-ingest")
    varrow(ax, col[3]+0.5, Y["Human Reviewer"]+0.4, Y["Truth DB / Exports"]+BH+0.2, "manual approve")

    # DB pipeline flow
    harrow(ax, col[4]-W/2-0.2, Y["Truth DB / Exports"]+1.5, col[3]+W/2+0.2, "INSERT zones")
    harrow(ax, col[3]-W/2-0.2, Y["Truth DB / Exports"]+1.5, col[2]+W/2+0.2, "API queries")

    plt.title("Zoning Worker Pipeline: PDF Extraction → Data Processing → Validation → Database", fontsize=16, pad=18)
    plt.savefig(output_png, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Saved: {output_png}\nSaved: {output_pdf}")

if __name__ == "__main__":
    draw()
