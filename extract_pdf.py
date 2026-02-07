
import PyPDF2
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def extract_content(pdf_path: str, output_path: str, max_pages: int = 5):
    """Extracts text content from a PDF file."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return

    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)
            num_to_extract = min(total_pages, max_pages)
            
            logger.info(f"Extracting {num_to_extract} of {total_pages} pages from {pdf_file.name}...")
            
            # Write mode to initialize file
            with open(output_path, 'w', encoding='utf-8') as out_f:
                for i in range(num_to_extract):
                    try:
                        text = reader.pages[i].extract_text()
                        out_f.write(f"--- Page {i+1} ---\n")
                        out_f.write(text if text else "[No text available on this page]")
                        out_f.write("\n\n")
                    except Exception as page_err:
                        logger.warning(f"Failed to extract page {i+1}: {page_err}")
                
        logger.info(f"Extraction successful. Content saved to: {output_path}")

    except Exception as e:
        logger.error(f"An error occurred during extraction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from PDF for analysis.")
    parser.add_argument("--input", default="d:\\AlphaGenome\\AphaGENOME.pdf", help="Path to input PDF")
    parser.add_argument("--output", default="d:\\AlphaGenome\\pdf_content.txt", help="Path to output text file")
    parser.add_argument("--pages", type=int, default=5, help="Maximum pages to extract")
    
    args = parser.parse_args()
    extract_content(args.input, args.output, args.pages)

