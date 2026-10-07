# Document Masking & AI-Powered Insights System

An AI-powered document processing system for detecting and masking Personally Identifiable Information (PII) using NER, Regex, OCR, OpenCV, and ONNX. The system provides a FastAPI backend and React-based frontend.

## Features

- PII detection using Named Entity Recognition (NER) and Regex
- Automated masking of sensitive information
- Face and ID-card detection and masking
- OCR-based text extraction
- Document sanitization for PDF, PPTX, and XLSX files
- Image processing using OpenCV
- FastAPI backend
- React frontend

## Technologies

- Python
- FastAPI
- React
- OpenCV
- NER
- Regex
- OCR
- ONNX

## Project Structure

```text
├── frontend/
│   ├── public/
│   └── src/
│       └── pages/
│
├── ai_insights.py
├── app.py
├── blur_photo_tool.py
├── masking.py
├── masking_tool.py
├── sanitize_pdf.py
├── sanitize_pptx.py
├── sanitize_xlsx.py
└── face_detection_yunet_2023mar.onnx
