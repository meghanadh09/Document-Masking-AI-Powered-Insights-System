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
```

## How to Run

### Backend

Make sure Python is installed on your system.

Start the FastAPI application:

```bash
uvicorn app:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

### Frontend

Navigate to the frontend directory:

```bash
cd frontend
```

Install the required dependencies:

```bash
npm install
```

Start the React application:

```bash
npm start
```

The frontend will be available at:

```text
http://localhost:3000
```

## Security

Do not upload confidential documents, API keys, passwords, environment files, or other sensitive information to the repository.

## Future Improvements

- Support for additional document formats
- Improved PII detection accuracy
- Enhanced document insights
- Additional masking and anonymization techniques
- Cloud-based deployment
Start the FastAPI application:

```bash
uvicorn app:app --reload
