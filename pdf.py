import os
import pdfkit
from PIL import Image
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas


from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# doc = SimpleDocTemplate("form_letter.pdf",pagesize=landscape(letter),
#                         rightMargin=72,leftMargin=72,
#                         topMargin=72,bottomMargin=18)

canvas = canvas.Canvas("result_4.pdf", pagesize=letter)
width,height = letter
print(width)
print(height)
canvas.setLineWidth(.3)
canvas.setFont('Helvetica', 12)
canvas.drawString(width/2-50,750,'MEDICAL REPORT')
#logo = "stage0.jpeg"


#canvas.drawImage(image_path, px from left border, px from bottom border, image width, image height)
#canvas.drawString(px from left border, px from bottom border, string)

canvas.drawImage( "stage0.jpeg", 50,500, 3*inch,3*inch) 
canvas.drawImage( "stage1.jpeg", width-50-(3*inch),500, 3*inch,3*inch) 
canvas.drawImage( "stage1.jpeg", 50,500-3*inch-50, 3*inch,3*inch) 
canvas.drawImage( "stage0.jpeg", width-50-(3*inch),500-3*inch-50, 3*inch,3*inch) 
c#anvas.drawString(140,500-3*inch-150,'Stage 1')
#canvas.drawString(width-(2*inch)-100,500-3*inch-150,'Stage 2')
canvas.drawString(120,500-3*inch-150,"LEFT EYE STAGE: ")
canvas.drawString(width-(2*inch)-100,500-3*inch-150,"RIGHT EYE STAGE: ")  



canvas.save()