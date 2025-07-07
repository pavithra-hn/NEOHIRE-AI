import io
from pdfminer.high_level import extract_text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import re

class PDFExtractor:
    def __init__(self):
        self.rsrcmgr = PDFResourceManager()
        self.laparams = LAParams()
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            # Method 1: Simple extraction
            text = extract_text(pdf_file)
            
            if not text.strip():
                # Method 2: More robust extraction
                text = self._extract_with_interpreter(pdf_file)
            
            return self._clean_text(text)
        
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")
    
    def _extract_with_interpreter(self, pdf_file):
        """Alternative extraction method for complex PDFs"""
        output_string = io.StringIO()
        device = TextConverter(self.rsrcmgr, output_string, laparams=self.laparams)
        interpreter = PDFPageInterpreter(self.rsrcmgr, device)
        
        pdf_file.seek(0)  # Reset file pointer
        
        for page in PDFPage.get_pages(pdf_file, check_extractable=True):
            interpreter.process_page(page)
        
        text = output_string.getvalue()
        device.close()
        output_string.close()
        
        return text
    
    def _clean_text(self, text):
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\-\.\@\(\)\+]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()