const current_file = path self

# converts a record of lists into a table
# precond: assumes that all lists are the same length
def "record columns to table" []: record -> table {
	let kv: table<key: string, value: list> = $in
		| transpose key value
	if ($kv | is-empty) {
		return []
	}
	let seed_col = $kv | first
	let seed = $seed_col
		| get value
		| enumerate
		| rename index ($seed_col | get key)
	$kv
		| slice 1..
		| reduce -f $seed {|col, table|
			let col_name = $col | get key
			let col_values = $col | get value
			$table
				| insert $col_name {|row| $col_values | get $row.index}
		}
		| reject index
}

# query searches for memories for the given query
export def query [--threshold: float]: string -> table<id: int score: float text: string> {
	let query = $in
	let threshold: float = $threshold | default 0.0

	http get $"($env.XRAG_ORIGIN)/query?q=($query | url encode)"
		| record columns to table
		| rename id text score
		| where score >= $threshold
		| sort-by score -r
}

# add creates a new memory and returns its id
export def add []: string -> int {
	http post --content-type "application/x-www-form-urlencoded" $"($env.XRAG_ORIGIN)/add" {
		memory: $in
	} | get 0
}

# relate creates a relationship between two memories
export def relate [child: int, parent: int, --type: string]: nothing -> nothing {
	let type: string = $type | default "related to"
	http post --content-type "application/x-www-form-urlencoded" $"($env.XRAG_ORIGIN)/relate" {
		child: $child
		parent: $parent
		rel_type: $type
	}
	null
}

# info finds related parent and children memories to the given memory
export def info [memory: int]: nothing -> record<parents: table<id: int, relationship: string, parent_content: string>, children: table<id: int, relationship: string, child_content: string>> {
	let results = http get $"($env.XRAG_ORIGIN)/info/($memory)"
	let parent_table: table<id: int, relationship: string, parent_content: string> = $results.parents
		| record columns to table
		| rename id relationship content
	let children_table: table<id: int, relationship: string, child_content: string> = $results.children
		| record columns to table
		| rename id relationship content
	{
		memory_content: $results.memory_content
		parents: $parent_table
		children: $children_table
	}
}
