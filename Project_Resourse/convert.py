import markdown
import pdfkit
import os
import sys
import glob

# CSS styling for clean PDF output
CSS_STYLE = """
<style>
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 40px auto;
    padding: 20px;
}
h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
    margin-top: 24px;
    margin-bottom: 16px;
    font-weight: 600;
}
h1 {
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}
h2 {
    border-bottom: 1px solid #bdc3c7;
    padding-bottom: 8px;
}
code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Consolas', monospace;
}
pre {
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 12px;
    overflow-x: auto;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}
th {
    background-color: #3498db;
    color: white;
}
tr:nth-child(even) {
    background-color: #f9f9f9;
}
blockquote {
    border-left: 4px solid #3498db;
    padding-left: 16px;
    margin-left: 0;
    color: #555;
}
a {
    color: #3498db;
    text-decoration: none;
}
</style>
"""

def find_wkhtmltopdf():
    """Find wkhtmltopdf executable on system"""
    # Common installation paths on Windows and Linux
    common_paths = [
        r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe',
        r'C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe',
        '/usr/bin/wkhtmltopdf',
        '/usr/local/bin/wkhtmltopdf',
    ]
    
    # Check common paths
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # Search in PATH
    try:
        import shutil
        path = shutil.which('wkhtmltopdf')
        if path:
            return path
    except:
        pass
    
    return None

def convert_md_to_pdf(md_file, wkhtmltopdf_path=None):
    """Convert a Markdown file to a clean PDF"""
    
    # Check if file exists
    if not os.path.exists(md_file):
        print(f"❌ Error: File '{md_file}' not found!")
        return False
    
    # Read the markdown content
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Convert Markdown to HTML
    html_content = markdown.markdown(text, extensions=['extra', 'tables', 'fenced_code', 'codehilite'])
    
    # Wrap HTML with CSS styling
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        {CSS_STYLE}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Generate output PDF filename
    # Remove .md extension if present, then add .pdf
    if md_file.endswith('.md'):
        output_pdf = md_file[:-3] + '.pdf'
    else:
        # For files like .md.resolved, just append .pdf
        output_pdf = md_file + '.pdf'
    
    # Configure pdfkit options
    options = {
        'encoding': 'UTF-8',
        'enable-local-file-access': None,
        'no-outline': None,
        'print-media-type': None,
    }
    
    # Set up pdfkit configuration with wkhtmltopdf path
    config = None
    if wkhtmltopdf_path:
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
    
    # Convert to PDF
    try:
        pdfkit.from_string(full_html, output_pdf, options=options, configuration=config)
        print(f"✅ PDF generated successfully: {output_pdf}")
        return True
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        print("\nMake sure wkhtmltopdf is installed and in your PATH.")
        return False

if __name__ == "__main__":
    # Find wkhtmltopdf executable
    wkhtmltopdf_path = find_wkhtmltopdf()
    
    if not wkhtmltopdf_path:
        print("❌ wkhtmltopdf not found!")
        print("\nPlease install wkhtmltopdf from: https://wkhtmltopdf.org/downloads.html")
        print("Or add it to your system PATH.")
        sys.exit(1)
    
    print(f"✓ Found wkhtmltopdf at: {wkhtmltopdf_path}\n")
    
    # Check if a file was provided as argument
    if len(sys.argv) > 1:
        md_file = sys.argv[1]
    else:
        # List available .md files
        md_files = [f for f in os.listdir('.') if f.endswith('.md')]
        
        if not md_files:
            print("❌ No .md files found in current directory!")
            sys.exit(1)
        
        print("📄 Available Markdown files:")
        for i, file in enumerate(md_files, 1):
            print(f"  {i}. {file}")
        
        # Let user choose
        try:
            choice = int(input("\nEnter the number of the file to convert (or 0 to convert all): "))
            if choice == 0:
                # Convert all files
                for file in md_files:
                    convert_md_to_pdf(file, wkhtmltopdf_path)
                sys.exit(0)
            elif 1 <= choice <= len(md_files):
                md_file = md_files[choice - 1]
            else:
                print("❌ Invalid choice!")
                sys.exit(1)
        except ValueError:
            print("❌ Invalid input!")
            sys.exit(1)
    
    # Convert the selected file
    convert_md_to_pdf(md_file, wkhtmltopdf_path)