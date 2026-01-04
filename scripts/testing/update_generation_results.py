"""
Update GENERATION_RESULTS.md with corrected circuit diagrams from DETAILED_CIRCUITS.txt
"""

import re

# Read DETAILED_CIRCUITS.txt
with open('docs/DETAILED_CIRCUITS.txt', 'r') as f:
    detailed_content = f.read()

# Read GENERATION_RESULTS.md
with open('GENERATION_RESULTS.md', 'r') as f:
    gen_results = f.read()

# Parse examples from DETAILED_CIRCUITS.txt
examples_section = []

# Split by example separators (the ---- lines)
example_blocks = detailed_content.split('\n' + '-'*80 + '\n')

for block in example_blocks:
    if not block.strip() or 'Example' not in block:
        continue

    lines = block.strip().split('\n')

    # Extract title (find line that starts with "Example")
    title = None
    title_line_idx = 0
    for i, line in enumerate(lines):
        title_match = re.match(r'Example \d+: (.+)', line)
        if title_match:
            title = title_match.group(1)
            title_line_idx = i
            break

    if not title:
        continue

    # Find all sections
    target_spec = []
    generated_circuit = []
    measured_perf = []
    netlist = []
    diagram = []
    analysis = []

    current_section = None
    in_code_block = False
    code_block_type = None

    for line in lines[title_line_idx + 1:]:
        if line.startswith('**Target Specification:**'):
            current_section = 'target'
            continue
        elif line.startswith('**Generated Circuit:**'):
            current_section = 'generated'
            continue
        elif line.startswith('**Measured Performance:**'):
            current_section = 'measured'
            continue
        elif line.startswith('**SPICE Netlist:**'):
            current_section = 'netlist'
            code_block_type = 'netlist'
            continue
        elif line.startswith('**Circuit Diagram:**'):
            current_section = 'diagram'
            code_block_type = 'diagram'
            continue
        elif line.startswith('**Analysis:**'):
            current_section = 'analysis'
            continue
        elif line.startswith('```'):
            in_code_block = not in_code_block
            if not in_code_block:
                code_block_type = None
            continue
        elif line.startswith('-'*80):
            break

        # Append content to appropriate section
        if current_section == 'target' and not in_code_block and line.strip():
            target_spec.append(line)
        elif current_section == 'generated' and not in_code_block and line.strip():
            generated_circuit.append(line)
        elif current_section == 'measured' and not in_code_block and line.strip():
            measured_perf.append(line)
        elif current_section == 'netlist' and in_code_block:
            netlist.append(line)
        elif current_section == 'diagram' and in_code_block:
            diagram.append(line)
        elif current_section == 'analysis' and not in_code_block and line.strip():
            analysis.append(line)

    # Build markdown section
    md_section = f"\n### {title}\n\n"
    md_section += "**Target Specification:**\n" + '\n'.join(target_spec) + "\n\n"
    md_section += "**Generated Circuit:**\n" + '\n'.join(generated_circuit) + "\n\n"
    md_section += "**Measured Performance:**\n" + '\n'.join(measured_perf) + "\n\n"
    md_section += "**SPICE Netlist:**\n```spice\n" + '\n'.join(netlist) + "\n```\n\n"
    md_section += "**Circuit Diagram:**\n```\n" + '\n'.join(diagram) + "\n```\n\n"
    md_section += "**Analysis:**\n" + '\n'.join(analysis) + "\n\n---"

    examples_section.append(md_section)

# Create the full examples section
full_examples = "\n## Detailed Test Examples (All 18 Test Cases)\n" + '\n'.join(examples_section) + "\n"

# Replace in GENERATION_RESULTS.md
# Find where the examples section starts and ends
start_marker = "## Detailed Test Examples (All 18 Test Cases)"
end_marker = "## Comparison: Before vs After KL Divergence"

start_idx = gen_results.find(start_marker)
end_idx = gen_results.find(end_marker)

if start_idx != -1 and end_idx != -1:
    new_content = gen_results[:start_idx] + full_examples + "\n" + gen_results[end_idx:]

    with open('GENERATION_RESULTS.md', 'w') as f:
        f.write(new_content)

    print("✅ Updated GENERATION_RESULTS.md with corrected circuit diagrams")
    print(f"   Processed {len(examples_section)} examples")
else:
    print("❌ Could not find example section markers in GENERATION_RESULTS.md")
