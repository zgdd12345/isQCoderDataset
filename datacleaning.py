#!/usr/bin/env python3
"""
Data cleaning module for quantum computing research papers.
Removes low signal-to-noise content (references, acknowledgments, image references)
and normalizes formatting to prepare high-quality data for LLM pretraining.
"""

import os
import re
import json
import argparse
import logging
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Any
from datetime import datetime


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""
    remove_references: bool = True
    remove_acknowledgments: bool = True
    remove_author_info: bool = True  # Remove author names, affiliations, emails
    remove_table_of_contents: bool = True  # Remove ToC/Contents sections
    remove_appendices: bool = False  # Keep appendices (valuable content)
    remove_images: bool = True  # Remove broken image references
    normalize_math: bool = True  # Normalize LaTeX spacing
    clean_tables: bool = True
    normalize_whitespace: bool = True
    output_format: str = "markdown"


def find_section_boundaries(content: str) -> Dict[str, Tuple[int, int]]:
    """
    Find boundaries of major sections in markdown content.

    Args:
        content: Markdown content as string

    Returns:
        Dict mapping section names to (start_line, end_line) tuples
    """
    lines = content.split('\n')
    sections = {}

    # Regex patterns for different section types
    references_pattern = re.compile(r'^#\s+References?$', re.IGNORECASE)
    acknowledgments_pattern = re.compile(r'^#\s+Acknowledg(e)?ments?$', re.IGNORECASE)
    appendix_pattern = re.compile(r'^#\s+Appendix\s+[A-Z0-9]', re.IGNORECASE)
    h1_pattern = re.compile(r'^#\s+(.+)$')

    for i, line in enumerate(lines):
        if references_pattern.match(line):
            sections['references'] = (i, len(lines))
        elif acknowledgments_pattern.match(line):
            # Find end of acknowledgments (next H1 section or end of file)
            end_line = len(lines)
            for j in range(i + 1, len(lines)):
                if h1_pattern.match(lines[j]):
                    end_line = j
                    break
            sections['acknowledgments'] = (i, end_line)
        elif appendix_pattern.match(line):
            # Track appendices but don't remove them (per user request)
            if 'appendices' not in sections:
                sections['appendices'] = []
            sections['appendices'].append((i, len(lines)))

    return sections


def remove_references(content: str) -> str:
    """
    Remove References section from content.

    Args:
        content: Markdown content

    Returns:
        Content with references removed
    """
    sections = find_section_boundaries(content)

    if 'references' not in sections:
        return content

    lines = content.split('\n')
    start, end = sections['references']

    # Check if acknowledgments come after references (unusual but possible)
    # If so, only remove from references to acknowledgments
    if 'acknowledgments' in sections:
        ack_start, _ = sections['acknowledgments']
        if ack_start > start:
            end = ack_start

    # Remove references section
    cleaned_lines = lines[:start] + lines[end:]

    return '\n'.join(cleaned_lines)


def remove_acknowledgments(content: str) -> str:
    """
    Remove Acknowledgments section from content.

    Args:
        content: Markdown content

    Returns:
        Content with acknowledgments removed
    """
    sections = find_section_boundaries(content)

    if 'acknowledgments' not in sections:
        return content

    lines = content.split('\n')
    start, end = sections['acknowledgments']

    # Remove acknowledgments section
    cleaned_lines = lines[:start] + lines[end:]

    return '\n'.join(cleaned_lines)


def remove_author_info(content: str) -> str:
    """
    Remove author information, affiliations, and contact details.
    This appears between the title (first H1) and Abstract section (or first H1 section after title).

    Args:
        content: Markdown content

    Returns:
        Content with author information removed
    """
    lines = content.split('\n')

    # Find the first H1 header (title) and the next meaningful section
    title_idx = -1
    next_section_idx = -1

    for i, line in enumerate(lines):
        # First H1 is the title
        if line.strip().startswith('# ') and title_idx == -1:
            title_idx = i
        # Find next H1 section (Abstract, Introduction, etc.) or paragraph starting with "Abstract."
        elif title_idx != -1 and next_section_idx == -1:
            # Check for H1 Abstract
            if re.match(r'^#\s+(Abstract|ABSTRACT|Introduction|INTRODUCTION|\d+\s+)', line.strip(), re.IGNORECASE):
                next_section_idx = i
                break
            # Check for paragraph starting with "Abstract." (non-header format)
            elif re.match(r'^Abstract\.', line.strip(), re.IGNORECASE):
                next_section_idx = i
                break
            # Check for Keywords line (sometimes appears before Abstract)
            elif re.match(r'^Keywords:', line.strip(), re.IGNORECASE):
                # Find start of this paragraph (go backwards to find the beginning)
                paragraph_start = i
                for j in range(i - 1, title_idx, -1):
                    if lines[j].strip() == '':
                        paragraph_start = j + 1
                        break
                next_section_idx = paragraph_start
                break

    # If we found both title and next section, remove everything in between
    if title_idx != -1 and next_section_idx != -1 and next_section_idx > title_idx + 1:
        # Keep title and next section, remove author info in between
        cleaned_lines = lines[:title_idx+1] + [''] + lines[next_section_idx:]
        return '\n'.join(cleaned_lines)

    return content


def remove_table_of_contents(content: str) -> str:
    """
    Remove Table of Contents section from content.
    Patterns: "# Contents" or "# Table of Contents"

    Args:
        content: Markdown content

    Returns:
        Content with table of contents removed
    """
    lines = content.split('\n')

    # Find Contents/Table of Contents section
    toc_start = -1
    toc_end = -1

    for i, line in enumerate(lines):
        # Match "# Contents" or "# Table of Contents"
        if re.match(r'^#\s+(Contents|Table\s+of\s+Contents)$', line.strip(), re.IGNORECASE):
            toc_start = i

            # Find the end of ToC - look for when actual content begins
            # ToC typically has numbered sections (1, 1.1, etc.) with page numbers
            # Actual content starts with a paragraph of substantial text
            for j in range(i + 1, len(lines)):
                stripped = lines[j].strip()

                # Skip empty lines
                if not stripped:
                    continue

                # If we find a substantial paragraph (>100 chars, not starting with number/section marker)
                # and not a heading, it's likely the start of actual content
                if len(stripped) > 100 and \
                   not re.match(r'^#', stripped) and \
                   not re.match(r'^\d+\.?\d*\s+', stripped) and \
                   not re.match(r'^[A-Z]\s+', stripped) and \
                   not re.match(r'^References', stripped, re.IGNORECASE):
                    toc_end = j
                    break

            # If we didn't find the end through paragraph detection,
            # look for max 100 lines or end of file
            if toc_end == -1:
                toc_end = min(i + 100, len(lines))

            break

    # Remove ToC if found
    if toc_start != -1 and toc_end != -1:
        cleaned_lines = lines[:toc_start] + lines[toc_end:]
        return '\n'.join(cleaned_lines)

    return content


def remove_image_references(content: str) -> str:
    """
    Remove image reference markdown syntax.
    Pattern: ![](images/hexhash.jpg)

    Args:
        content: Markdown content

    Returns:
        Content with image references removed
    """
    # Match image syntax: ![](images/hash.ext) or ![alt](images/hash.ext)
    image_pattern = re.compile(r'!\[.*?\]\(images/[a-f0-9]+\.(jpg|png|gif|pdf)\)', re.IGNORECASE)

    # Remove all image references
    cleaned = image_pattern.sub('', content)

    # Also remove common figure caption patterns if they're now orphaned
    # Look for "Figure N:" at the start of a line that's now alone
    figure_caption_pattern = re.compile(r'^\s*Figure\s+\d+:.*?$', re.MULTILINE)
    cleaned = figure_caption_pattern.sub('', cleaned)

    return cleaned


def normalize_math_spacing(content: str) -> str:
    """
    Normalize LaTeX math spacing from PDF conversion artifacts.

    Args:
        content: Markdown content

    Returns:
        Content with normalized math spacing
    """
    # Fix common spacing issues in LaTeX
    # Pattern: "$ ." -> "$."
    content = re.sub(r'\$\s+\.', '$.', content)

    # Pattern: "$ ," -> "$,"
    content = re.sub(r'\$\s+,', '$,', content)

    # Pattern: "$ ;" -> "$;"
    content = re.sub(r'\$\s+;', '$;', content)

    # Pattern: "$ )" -> "$)"
    content = re.sub(r'\$\s+\)', '$)', content)

    # Pattern: "( $" -> "($"
    content = re.sub(r'\(\s+\$', '($', content)

    # Pattern: "[ $" -> "[$"
    content = re.sub(r'\[\s+\$', '[$', content)

    # Pattern: "{ $" -> "{$"
    content = re.sub(r'\{\s+\$', '{$', content)

    return content


def normalize_whitespace(content: str) -> str:
    """
    Normalize whitespace in content.
    - Max 2 consecutive newlines
    - Remove trailing whitespace from lines
    - Ensure file ends with single newline

    Args:
        content: Markdown content

    Returns:
        Content with normalized whitespace
    """
    # Split into lines, strip trailing whitespace from each
    lines = [line.rstrip() for line in content.split('\n')]

    # Join back and replace 3+ newlines with exactly 2
    content = '\n'.join(lines)
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Ensure single newline at end
    content = content.rstrip() + '\n'

    return content


def clean_paper(input_path: str, output_path: str, config: CleaningConfig) -> Dict[str, Any]:
    """
    Clean a single paper file.

    Args:
        input_path: Path to input markdown file
        output_path: Path to output cleaned file
        config: CleaningConfig instance

    Returns:
        Statistics dict with cleaning metrics
    """
    try:
        # Read file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_size = len(content)
        original_lines = content.count('\n')

        stats = {
            'file': os.path.basename(input_path),
            'original_size': original_size,
            'original_lines': original_lines,
            'references_removed': False,
            'acknowledgments_removed': False,
            'author_info_removed': False,
            'toc_removed': False,
            'images_removed': False,
        }

        # Apply cleaning operations in order
        # Remove author info first (before other sections)
        if config.remove_author_info:
            new_content = remove_author_info(content)
            if len(new_content) < len(content):
                stats['author_info_removed'] = True
                content = new_content

        # Remove table of contents
        if config.remove_table_of_contents:
            new_content = remove_table_of_contents(content)
            if len(new_content) < len(content):
                stats['toc_removed'] = True
                content = new_content

        if config.remove_acknowledgments:
            # Remove acknowledgments first (usually appears before references)
            new_content = remove_acknowledgments(content)
            if len(new_content) < len(content):
                stats['acknowledgments_removed'] = True
                content = new_content

        if config.remove_references:
            new_content = remove_references(content)
            if len(new_content) < len(content):
                stats['references_removed'] = True
                content = new_content

        if config.remove_images:
            new_content = remove_image_references(content)
            if len(new_content) < len(content):
                stats['images_removed'] = True
                content = new_content

        if config.normalize_math:
            content = normalize_math_spacing(content)

        if config.normalize_whitespace:
            content = normalize_whitespace(content)

        # Calculate final stats
        final_size = len(content)
        final_lines = content.count('\n')

        stats['final_size'] = final_size
        stats['final_lines'] = final_lines
        stats['size_reduction_bytes'] = original_size - final_size
        stats['size_reduction_pct'] = (1 - final_size / original_size) * 100 if original_size > 0 else 0
        stats['lines_reduction'] = original_lines - final_lines

        # Write cleaned file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        stats['success'] = True

    except Exception as e:
        logger.error(f"Error cleaning {input_path}: {e}")
        stats = {
            'file': os.path.basename(input_path),
            'success': False,
            'error': str(e)
        }

    return stats


def clean_dataset(
    input_dir: str,
    output_dir: str,
    config: CleaningConfig,
    sample_size: int = 0,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Clean all markdown files in a directory.

    Args:
        input_dir: Input directory containing markdown files
        output_dir: Output directory for cleaned files
        config: CleaningConfig instance
        sample_size: If >0, only process this many random files
        dry_run: If True, don't write files, just calculate stats

    Returns:
        Aggregate statistics dict
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all markdown files
    md_files = list(input_path.glob('*.md'))
    total_files = len(md_files)

    logger.info(f"Found {total_files} markdown files in {input_dir}")

    # Sample if requested
    if sample_size > 0 and sample_size < total_files:
        md_files = random.sample(md_files, sample_size)
        logger.info(f"Sampling {sample_size} files for processing")

    # Process files
    all_stats = []
    successful = 0
    failed = 0

    for i, file_path in enumerate(md_files, 1):
        logger.info(f"Processing {i}/{len(md_files)}: {file_path.name}")

        if dry_run:
            # Just read and calculate what would happen
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"  Would clean: {file_path.name} ({len(content)} bytes)")
            except Exception as e:
                logger.error(f"  Error reading {file_path.name}: {e}")
            continue

        # Create output path
        out_file = output_path / file_path.name

        # Clean the file
        file_stats = clean_paper(str(file_path), str(out_file), config)
        all_stats.append(file_stats)

        if file_stats.get('success', False):
            successful += 1
            logger.info(f"  ✓ Cleaned: {file_stats['size_reduction_pct']:.1f}% reduction "
                       f"({file_stats['size_reduction_bytes']} bytes)")
        else:
            failed += 1

    if dry_run:
        logger.info("Dry run complete - no files were written")
        return {}

    # Calculate aggregate statistics
    aggregate_stats = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'total_files': len(md_files),
        'successful': successful,
        'failed': failed,
        'config': asdict(config),
        'totals': {
            'original_size': sum(s.get('original_size', 0) for s in all_stats if s.get('success')),
            'final_size': sum(s.get('final_size', 0) for s in all_stats if s.get('success')),
            'original_lines': sum(s.get('original_lines', 0) for s in all_stats if s.get('success')),
            'final_lines': sum(s.get('final_lines', 0) for s in all_stats if s.get('success')),
        },
        'counts': {
            'references_removed': sum(1 for s in all_stats if s.get('references_removed', False)),
            'acknowledgments_removed': sum(1 for s in all_stats if s.get('acknowledgments_removed', False)),
            'author_info_removed': sum(1 for s in all_stats if s.get('author_info_removed', False)),
            'toc_removed': sum(1 for s in all_stats if s.get('toc_removed', False)),
            'images_removed': sum(1 for s in all_stats if s.get('images_removed', False)),
        },
        'file_stats': all_stats
    }

    # Calculate percentages
    if aggregate_stats['totals']['original_size'] > 0:
        aggregate_stats['totals']['size_reduction_pct'] = (
            (1 - aggregate_stats['totals']['final_size'] / aggregate_stats['totals']['original_size']) * 100
        )
        aggregate_stats['totals']['size_reduction_bytes'] = (
            aggregate_stats['totals']['original_size'] - aggregate_stats['totals']['final_size']
        )

    # Save stats to JSON
    stats_file = output_path / 'stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(aggregate_stats, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Cleaning complete!")
    logger.info(f"  Processed: {successful}/{len(md_files)} files")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Original size: {aggregate_stats['totals']['original_size']:,} bytes "
               f"({aggregate_stats['totals']['original_size']/1024/1024:.1f} MB)")
    logger.info(f"  Final size: {aggregate_stats['totals']['final_size']:,} bytes "
               f"({aggregate_stats['totals']['final_size']/1024/1024:.1f} MB)")
    logger.info(f"  Reduction: {aggregate_stats['totals'].get('size_reduction_pct', 0):.1f}% "
               f"({aggregate_stats['totals'].get('size_reduction_bytes', 0):,} bytes)")
    logger.info(f"  References removed: {aggregate_stats['counts']['references_removed']} files")
    logger.info(f"  Acknowledgments removed: {aggregate_stats['counts']['acknowledgments_removed']} files")
    logger.info(f"  Author info removed: {aggregate_stats['counts']['author_info_removed']} files")
    logger.info(f"  Table of contents removed: {aggregate_stats['counts']['toc_removed']} files")
    logger.info(f"  Images removed: {aggregate_stats['counts']['images_removed']} files")
    logger.info(f"  Statistics saved to: {stats_file}")
    logger.info(f"{'='*60}\n")

    return aggregate_stats


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='Clean quantum computing research papers for LLM pretraining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean all files in data/raw
  python datacleaning.py --input-dir data/raw --output-dir data/cleaned

  # Test on 10 random files
  python datacleaning.py --input-dir data/raw --output-dir data/test --sample-size 10

  # Dry run to preview changes
  python datacleaning.py --dry-run

  # Keep image references and skip math normalization
  python datacleaning.py --keep-images --no-normalize-math
        """
    )

    parser.add_argument(
        '--input-dir',
        default='data/raw',
        help='Input directory containing markdown files (default: data/raw)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/cleaned',
        help='Output directory for cleaned files (default: data/cleaned)'
    )
    parser.add_argument(
        '--keep-author-info',
        action='store_true',
        help='Keep author information and affiliations (default: remove)'
    )
    parser.add_argument(
        '--keep-images',
        default=False,
        action='store_true',
        help='Keep image references (default: remove)'
    )
    parser.add_argument(
        '--no-normalize-math',
        action='store_true',
        help='Skip LaTeX spacing normalization (default: normalize)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=0,
        help='Process only N random files for testing (default: all files)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be cleaned without writing files'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Generate statistics report only (requires existing cleaned output)'
    )

    args = parser.parse_args()

    # Create configuration
    config = CleaningConfig(
        remove_references=True,
        remove_acknowledgments=True,
        remove_author_info=not args.keep_author_info,
        remove_table_of_contents=True,
        remove_appendices=False,
        remove_images=not args.keep_images,
        normalize_math=not args.no_normalize_math,
        clean_tables=True,
        normalize_whitespace=True,
    )

    logger.info("Data Cleaning Configuration:")
    logger.info(f"  Remove references: {config.remove_references}")
    logger.info(f"  Remove acknowledgments: {config.remove_acknowledgments}")
    logger.info(f"  Remove author info: {config.remove_author_info}")
    logger.info(f"  Remove table of contents: {config.remove_table_of_contents}")
    logger.info(f"  Remove appendices: {config.remove_appendices}")
    logger.info(f"  Remove images: {config.remove_images}")
    logger.info(f"  Normalize math: {config.normalize_math}")
    logger.info(f"  Normalize whitespace: {config.normalize_whitespace}")
    logger.info(f"  Input: {args.input_dir}")
    logger.info(f"  Output: {args.output_dir}")
    if args.sample_size > 0:
        logger.info(f"  Sample size: {args.sample_size}")
    logger.info("")

    # Run cleaning
    clean_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=config,
        sample_size=args.sample_size,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
