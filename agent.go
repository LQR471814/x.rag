package main

/*
# Structures

Claim:
- text: The claim text.
- evidence:
	- other claims
	- text from an external source

# Agents

Reader:
- scroll_down -> scroll to the next text chunk
- add -> add claims

Organizer:
- Go through all the claims on the current layer.
- See if the current claim supports any of them.
- If current can be considered a more specific case of the claim being
compared:
	- Set the evidence for the claim as the current layer and recursively run
	organizer with the same claim
- If there is a commonality between the two claims being compared:
	-
- If the current claim is not a more specific case of any claims on the current
layer:
	- Add the current claim to the current layer
*/
