"""
Add labels to GENERATION_RESULTS.md indicating which topologies were seen in training.
"""

import re

# Training topologies from novelty analysis
TRAINING_TOPOLOGIES = {'1R+1C', '2R+1C+1L', '1R+1C+1L', '4R+1C+1L'}

# Read the GENERATION_RESULTS.md file
with open('GENERATION_RESULTS.md', 'r') as f:
    content = f.read()

# Split into lines for processing
lines = content.split('\n')

# Process each line
updated_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    updated_lines.append(line)

    # Check if this is a topology line
    topology_match = re.match(r'^- Topology: \d+ edges? \(([^)]+)\)$', line)

    if topology_match:
        topology = topology_match.group(1)

        # Check if the next line is already a topology label (to avoid duplicates)
        if i + 1 < len(lines) and '**Topology Status:**' in lines[i + 1]:
            # Skip the next line (old label)
            i += 1
        else:
            # Add the label
            if topology in TRAINING_TOPOLOGIES:
                updated_lines.append(f"- **Topology Status:** ✅ Seen in training data")
            else:
                updated_lines.append(f"- **Topology Status:** ❌ NOVEL (not in training) - Invalid!")

    i += 1

# Join back together
updated_content = '\n'.join(updated_lines)

# Write back to file
with open('GENERATION_RESULTS.md', 'w') as f:
    f.write(updated_content)

print("✅ Added topology labels to GENERATION_RESULTS.md")
print("\nTraining topologies:")
for topo in sorted(TRAINING_TOPOLOGIES):
    print(f"  - {topo}")
print("\nNovel topologies will be marked as INVALID")
