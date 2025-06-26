import pdfplumber

def extract_text_from_pdf(path):
    full_text = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text()
                if text:
                    full_text.append(text) 
            except Exception as e:
                print(f"Error extracting text from page {i}: {e}")
        return "\n".join(full_text)

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks