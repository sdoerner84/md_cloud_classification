'''
Created on 07.04.2025

@author: steffen.ziegler
'''
import textwrap
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def add_text_page(pdfstream: PdfPages, text: str, chars_per_line: int=100,
                  lines_per_page: int=49, fontsize: int=10):
    """
    Add text to a PDF stream that is automatically wrapped in lines and pages.

    @pdfstream
    @text
    @chars_per_line (default=100) number of characters per line
    @lines_per_page (default=40) number of lines per page
    @fontsize
    """
    # Wrap text into lines
    wrapped_lines = []
    for paragraph in text.split('\n'):
        if paragraph.strip() == '':
            wrapped_lines.append('')
            continue
        wrapped_lines.extend(textwrap.wrap(paragraph, width=chars_per_line))

    # Wrap lines into pages
    for i in range(0, len(wrapped_lines), lines_per_page):
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size in inches
        ax.axis('off')
        ax.set_position([0.01, 0, 1, 0.99])
        page_lines = wrapped_lines[i:i+lines_per_page]
        for j, line in enumerate(page_lines):
            ax.text(0.01, 1 - 0.01 - j * 0.02, line, fontsize=fontsize,
                    va='top', transform=ax.transAxes)
        pdfstream.savefig(fig)
        plt.close(fig)
