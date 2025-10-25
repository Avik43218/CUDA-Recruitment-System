import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Extract essential information

def extract_experience_years(text: str) -> str:
    # Extract number of years of experience from text.
    match = re.search(r'(\d+)\s*(?:\+)?\s*(?:years|yrs)', text, re.IGNORECASE)
    return match.group(1) if match else "0"

def extract_programming_languages(text: str) -> list:
    langs = [
        "python", "java", "c++", "c#", "javascript", "typescript",
        "go", "ruby", "php", "swift", "kotlin", "rust", "matlab",
        "r", "scala", "perl", "dart", "css", "html", "xml"
    ]
    return sorted({lang for lang in langs if re.search(rf"\b{lang}\b", text, re.IGNORECASE)})

def extract_tools(text: str) -> list:
    tools = [
        "git", "docker", "kubernetes", "jenkins", "ansible", "terraform",
        "jira", "postman", "aws", "gcp", "azure", "visual studio", "eclipse",
        "intellij", "notion", "slack", "excel"
    ]
    return sorted({tool for tool in tools if re.search(rf"\b{tool}\b", text, re.IGNORECASE)})

def extract_frameworks(text: str) -> list:
    frameworks = [
        "django", "flask", "react", "angular", "vue", "spring", "spring boot",
        "fastapi", "express", "laravel", "dotnet", ".net", "tensorflow",
        "pytorch", "keras", "bootstrap", "node.js"
    ]
    return sorted({fw for fw in frameworks if re.search(rf"\b{fw}\b", text, re.IGNORECASE)})

def extract_databases(text: str) -> list:
    databases = ["mysql", "mongodb", "postgresql", "sqlite"]
    return sorted({db for db in databases if re.search(rf"\b{db}\b", text, re.IGNORECASE)})

# Cleaning the resumes

def clean_resume_text(text: str) -> str:
    # Normalize whitespace and lowercase text.
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def process_resume(filepath: Path, output_dir: Path) -> str:
    # Extract and save cleaned information from one resume.
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        clean_text = clean_resume_text(raw_text)

        exp_years = extract_experience_years(clean_text)
        langs = extract_programming_languages(clean_text)
        tools = extract_tools(clean_text)
        frameworks = extract_frameworks(clean_text)
        databases = extract_databases(clean_text)

        # Prepare cleaned content
        output_lines = [
            f"experience_years: {exp_years}",
            "\nprogramming_languages:",
            *langs,
            "\ntools:",
            *tools,
            "\nframeworks:",
            *frameworks,
            "\ndatabases:",
            *databases,
        ]

        output_path = output_dir / f"cleaned_{filepath.stem}.txt"
        with open(output_path, "w", encoding="utf-8") as out:
            out.write("\n".join(output_lines))

        return f"Cleaned: {filepath.name}"

    except Exception as e:
        return f"Error processing {filepath.name}: {e}"


def main():
    input_dir = "D:\\Programs\\Hackathon\\parallel-recruition-system\\sample-data"
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"Error: Folder '{input_dir}' does not exist.")
        return

    output_dir = Path("cleaned_resumes")
    output_dir.mkdir(exist_ok=True)

    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        print("No .txt resumes found.")
        return

    print(f"Processing {len(txt_files)} resumes using parallel execution...\n")

    # Use all available CPU cores
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_resume, file, output_dir): file for file in txt_files}
        for future in as_completed(futures):
            print(future.result())

    print("\nAll resumes processed successfully!")

if __name__ == "__main__":
    main()
