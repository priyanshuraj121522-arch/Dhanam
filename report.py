from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

def build_pdf(path, title, metrics, charts):
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=24,leftMargin=24,topMargin=24,bottomMargin=24)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"<b>{title}</b>", styles['Title']))
    story.append(Spacer(1,12))

    # Metrics table
    data = [["Metric","Value"]] + [[k,str(v)] for k,v in metrics.items()]
    tbl = Table(data, hAlign='LEFT', colWidths=[200, 200])
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.black),
                             ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                             ('GRID',(0,0),(-1,-1),0.25,colors.grey),
                             ('BACKGROUND',(0,1),(-1,-1),colors.whitesmoke)]))
    story.append(tbl)
    story.append(Spacer(1,12))

    for label, img_path in charts:
        story.append(Paragraph(label, styles['Heading3']))
        story.append(Image(img_path, width=480, height=260))
        story.append(Spacer(1,12))

    doc.build(story)
