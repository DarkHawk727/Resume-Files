import argparse
import asyncio
import logging
import os
import subprocess
import sys

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_openai import OpenAI
from langchain.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prompt = PromptTemplate(
    input_variables=["job_listing_text", "resume_text"],
    template=f"""
    Please write a cover letter formatted in markdown for the following job listing and resume. \
    Give me only the letter, no preamble or any other text. DO NOT format the markdown within triple backticks. \
    Use only experiences, projects, and skills that are directly contained in the resume. \
    The general format should contain a paragraph talking about the target company, the next should be about how my experience would be an asset to the company. \
    The final should be how my ClifftonStrengths Finder results would make me good cultural fit. The overall length should not exceed 500 words. DO NOT include any information like my email or address in the header of the letter.:\n\n

    Job Listing:
    {job_listing_text}\n\n

    Resume:
    {resume_text}\n\n
    """,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a cover letter from a job listing URL and a resume."
    )
    parser.add_argument("url", help="URL of the job listing")
    return parser.parse_args()


async def extract_text(url: str) -> str:
    """Asynchronously extract text from the given URL.

    Args:
        url (str): URL to extract text from.

    Returns:
        str: Extracted text content.
    """
    try:
        loader = AsyncHtmlLoader(web_path=url)
        docs = await loader.load()
        html2text = Html2TextTransformer()
        docs = html2text.transform_documents(documents=docs)
        return docs[0].page_content
    except Exception as e:
        logger.error(f"Failed to extract text from the URL: {e}")
        sys.exit(1)


async def main() -> None:
    """Main function to generate a cover letter from a job listing URL and a resume."""
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    try:
        with open("main.tex", "r") as f:
            resume_text = f.read()
    except Exception as e:
        logger.error(f"Failed to read resume file: {e}")
        sys.exit(1)

    job_listing_text = await extract_text(args.url)

    llm = OpenAI(model="gpt-4", api_key=api_key, max_tokens=1000)

    chain = prompt | llm | StrOutputParser()
    chain.ainvoke()

    with open("coverletter.md", "w") as md_file:
        md_file.write(cover_letter)

    try:
        subprocess.run(
            ["pandoc", "coverletter.md", "-o", "coverletter.pdf"], check=True
        )
        print("Successfully converted Markdown to PDF.")
    except subprocess.CalledProcessError:
        print("Error occurred while converting Markdown to PDF.")


if __name__ == "__main__":
    asyncio.run(main())
