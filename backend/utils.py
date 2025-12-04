import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import io
import cantools
from lxml import etree
import ast

def process_file(file_content: bytes, filename: str) -> List[Document]:
    """
    Process a file (PDF, Excel, XML, DBC, CDD, Python) and return a list of Documents.
    """
    file_ext = filename.split('.')[-1].lower()
    documents = []

    if file_ext == 'pdf':
        # Process PDF files
        import pypdf
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_content))
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                documents.append(Document(page_content=page_text, metadata={"source": filename, "page": i+1}))

    elif file_ext in ['xlsx', 'xls']:
        # Process Excel files
        df = pd.read_excel(io.BytesIO(file_content))
        for index, row in df.iterrows():
            content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            documents.append(Document(page_content=content, metadata={"source": filename, "row": index+1}))

    elif file_ext == 'xml':
        # Process XML files (generic XML and ARXML/CDD)
        documents.extend(process_xml_file(file_content, filename))

    elif file_ext == 'dbc':
        # Process DBC (CAN database) files
        documents.extend(process_dbc_file(file_content, filename))

    elif file_ext in ['arxml', 'cdd', 'a2l']:
        # Process ARXML/CDD/A2L files (automotive calibration data)
        documents.extend(process_automotive_file(file_content, filename, file_ext))

    elif file_ext == 'py':
        # Process Python files
        documents.extend(process_python_file(file_content, filename))

    return documents

def process_xml_file(file_content: bytes, filename: str) -> List[Document]:
    """Process XML files and extract structured information."""
    documents = []
    try:
        root = etree.fromstring(file_content)

        # Extract all elements with text content
        def extract_elements(element, path=""):
            current_path = f"{path}/{element.tag}" if path else element.tag

            # Get element text and attributes
            text_parts = []
            if element.text and element.text.strip():
                text_parts.append(f"Text: {element.text.strip()}")

            if element.attrib:
                attrs = " | ".join([f"{k}={v}" for k, v in element.attrib.items()])
                text_parts.append(f"Attributes: {attrs}")

            if text_parts:
                content = f"Element: {current_path}\n" + "\n".join(text_parts)
                documents.append(Document(
                    page_content=content,
                    metadata={"source": filename, "element_path": current_path, "type": "xml"}
                ))

            # Process children
            for child in element:
                extract_elements(child, current_path)

        extract_elements(root)

    except Exception as e:
        # Fallback: treat as plain text
        text = file_content.decode('utf-8', errors='ignore')
        documents.append(Document(
            page_content=f"XML Content:\n{text}",
            metadata={"source": filename, "type": "xml", "parsing_error": str(e)}
        ))

    return documents

def process_dbc_file(file_content: bytes, filename: str) -> List[Document]:
    """Process DBC (CAN database) files and extract CAN message definitions."""
    documents = []
    try:
        # Parse DBC file
        db = cantools.database.load_string(file_content.decode('utf-8', errors='ignore'))

        # Extract messages
        for message in db.messages:
            content_parts = [
                f"Message Name: {message.name}",
                f"CAN ID: 0x{message.frame_id:X} ({message.frame_id})",
                f"Length: {message.length} bytes",
                f"Cycle Time: {message.cycle_time}ms" if message.cycle_time else "Cycle Time: N/A",
                f"Comment: {message.comment}" if message.comment else ""
            ]

            # Extract signals
            signal_info = []
            for signal in message.signals:
                sig_info = [
                    f"  Signal: {signal.name}",
                    f"    Start Bit: {signal.start}, Length: {signal.length} bits",
                    f"    Type: {signal.is_signed and 'Signed' or 'Unsigned'}",
                    f"    Scale: {signal.scale}, Offset: {signal.offset}",
                    f"    Min: {signal.minimum}, Max: {signal.maximum}",
                    f"    Unit: {signal.unit}" if signal.unit else "",
                    f"    Comment: {signal.comment}" if signal.comment else ""
                ]
                signal_info.append("\n".join([s for s in sig_info if s]))

            if signal_info:
                content_parts.append("\nSignals:")
                content_parts.extend(signal_info)

            content = "\n".join([p for p in content_parts if p])
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": filename,
                    "message_name": message.name,
                    "can_id": f"0x{message.frame_id:X}",
                    "type": "dbc"
                }
            ))

    except Exception as e:
        # Fallback: treat as plain text
        text = file_content.decode('utf-8', errors='ignore')
        documents.append(Document(
            page_content=f"DBC Content:\n{text}",
            metadata={"source": filename, "type": "dbc", "parsing_error": str(e)}
        ))

    return documents

def process_automotive_file(file_content: bytes, filename: str, file_ext: str) -> List[Document]:
    """Process ARXML, CDD, or A2L automotive calibration files."""
    documents = []
    try:
        if file_ext in ['arxml', 'cdd']:
            # Parse as XML (ARXML/CDD are XML-based)
            root = etree.fromstring(file_content)

            # Look for common AUTOSAR/calibration elements
            # Extract SW-COMPONENTS, CALIBRATION-PARAMETERS, etc.
            namespaces = root.nsmap

            # Find all calibration parameters
            for elem in root.iter():
                tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

                # Focus on important elements
                if any(keyword in tag_name.upper() for keyword in ['PARAMETER', 'VARIABLE', 'COMPONENT', 'SIGNAL', 'INTERFACE']):
                    content_parts = [f"Element Type: {tag_name}"]

                    if elem.text and elem.text.strip():
                        content_parts.append(f"Value: {elem.text.strip()}")

                    if elem.attrib:
                        for key, val in elem.attrib.items():
                            content_parts.append(f"{key}: {val}")

                    # Get child elements
                    for child in elem:
                        child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                        if child.text and child.text.strip():
                            content_parts.append(f"{child_tag}: {child.text.strip()}")

                    if len(content_parts) > 1:  # Only add if we have more than just the type
                        documents.append(Document(
                            page_content="\n".join(content_parts),
                            metadata={"source": filename, "element_type": tag_name, "type": file_ext}
                        ))

        elif file_ext == 'a2l':
            # A2L files are text-based, parse as structured text
            text = file_content.decode('utf-8', errors='ignore')
            # Split by major sections
            sections = text.split('/begin')
            for section in sections[1:]:  # Skip first empty split
                section_name = section.split()[0] if section.split() else "UNKNOWN"
                content = f"/begin {section_name}\n{section[:1000]}"  # First 1000 chars
                documents.append(Document(
                    page_content=content,
                    metadata={"source": filename, "section": section_name, "type": "a2l"}
                ))

    except Exception as e:
        # Fallback
        text = file_content.decode('utf-8', errors='ignore')
        documents.append(Document(
            page_content=f"{file_ext.upper()} Content:\n{text[:2000]}",
            metadata={"source": filename, "type": file_ext, "parsing_error": str(e)}
        ))

    return documents

def process_python_file(file_content: bytes, filename: str) -> List[Document]:
    """Process Python files and extract classes, functions, and docstrings."""
    documents = []
    try:
        code = file_content.decode('utf-8', errors='ignore')
        tree = ast.parse(code)

        # Extract module-level docstring
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            documents.append(Document(
                page_content=f"Module Docstring:\n{module_docstring}",
                metadata={"source": filename, "type": "python", "element_type": "module"}
            ))

        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = [
                    f"Function: {node.name}",
                    f"Arguments: {', '.join([arg.arg for arg in node.args.args])}",
                ]

                docstring = ast.get_docstring(node)
                if docstring:
                    func_info.append(f"Docstring: {docstring}")

                # Get function body as text (simplified)
                func_line = f"Line: {node.lineno}"
                func_info.append(func_line)

                documents.append(Document(
                    page_content="\n".join(func_info),
                    metadata={
                        "source": filename,
                        "type": "python",
                        "element_type": "function",
                        "name": node.name,
                        "line": node.lineno
                    }
                ))

            elif isinstance(node, ast.ClassDef):
                class_info = [
                    f"Class: {node.name}",
                    f"Base Classes: {', '.join([base.id for base in node.bases if isinstance(base, ast.Name)])}",
                ]

                docstring = ast.get_docstring(node)
                if docstring:
                    class_info.append(f"Docstring: {docstring}")

                # List methods
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                if methods:
                    class_info.append(f"Methods: {', '.join(methods)}")

                documents.append(Document(
                    page_content="\n".join(class_info),
                    metadata={
                        "source": filename,
                        "type": "python",
                        "element_type": "class",
                        "name": node.name,
                        "line": node.lineno
                    }
                ))

        # If no structured content found, include the whole file
        if not documents:
            documents.append(Document(
                page_content=f"Python Code:\n{code}",
                metadata={"source": filename, "type": "python"}
            ))

    except Exception as e:
        # Fallback: treat as plain text
        text = file_content.decode('utf-8', errors='ignore')
        documents.append(Document(
            page_content=f"Python File Content:\n{text}",
            metadata={"source": filename, "type": "python", "parsing_error": str(e)}
        ))

    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)
