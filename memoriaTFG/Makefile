# LaTeX document compilation Makefile

# Configuration
LATEX = pdflatex
BIBTEX = bibtex

# Default target: build the PDF
all: force_rebuild main.pdf

# Phony target to force rebuild
force_rebuild:
	@if [ -f main.pdf ]; then \
		echo "Eliminando PDF existente..."; \
		rm -f main.pdf; \
	fi

# Rule to build the PDF
main.pdf: main.tex
	$(LATEX) main
	$(BIBTEX) main
	$(LATEX) main
	$(LATEX) main
	@echo "PDF generado correctamente: main.pdf"

# Clean temporary files
clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg main.pdf

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Elimina el PDF existente y construye uno nuevo (default)"
	@echo "  clean    - Elimina archivos auxiliares"
	@echo "  cleanall - Elimina archivos auxiliares y el PDF"
	@echo "  help     - Muestra este mensaje de ayuda"

.PHONY: all clean cleanall help force_rebuild