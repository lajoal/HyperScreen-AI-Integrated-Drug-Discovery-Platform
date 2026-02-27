from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import Table
import os

def generate_report(df, output="Final_Report.pdf"):

    doc = SimpleDocTemplate(output)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("HyperScreen Final Candidate Report", styles['Title']))
    elements.append(Spacer(1,12))

    data = [["Compound", "Vina", "CNN", "Toxicity", "Composite", "MD RMSD"]]

    for _, row in df.head(10).iterrows():
        data.append([
            row["compound"],
            row["vina_score"],
            row["cnn_score"],
            row["tox_score"],
            row["composite_score"],
            row["md_rmsd"]
        ])

    table = Table(data)
    elements.append(table)

    doc.build(elements)
    return output
