package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/ollama/ollama/api"
)

const gen_model = "gemma3:1b-it-qat"
const embed_model = "nomic-embed-text"
const embed_dim = 768

func main() {
	err := run_cmp()
	if err != nil {
		panic(err)
	}
}

const docs = `
<docs>
*api.txt*               Nvim


                 NVIM REFERENCE MANUAL    by Thiago de Arruda


Nvim API                                                           *API* *api*

Nvim exposes a powerful API that can be used by plugins and external processes
via |RPC|, |Lua| and Vimscript (|eval-api|).

Applications can also embed libnvim to work with the C API directly.

                                      Type |gO| to see the table of contents.
</docs>

Using the docs `

// const prompt = `Your task is to summarize a list of claims from the text. Your response should be in the format { claims: string[] }.
//
// %s`

// Reader
// Operations:
// - scroll_down -> scroll to the next text chunk
// - add -> add claims

// Between a claim and evidence, there is the bonding element that you have to consider.
// Exactly how reliable can we say a claim derives from evidence?
// Summarizing an existing established text is likely to be reliable.
// Summarizing a list of summaries to give an overview of a field is also likely to be reliable.
// That is:
//  - The process of summarization is reliable, it is hard to mess up.
//  - The process of abstracting from data is not necessarily reliable however.
//
// Summarization is not really abstraction, it is just rewriting the same text in a different format while omitting certain claims.
// The ideal claim derivation would result in the *exact same* information as the original text, just in a cleaner format.
// So summarization and claim derivation should be treated as different processes.

// Claim:
// - Text: The text that makes up the claim. (the generic concept or abstraction)
// - Evidence:
//   - Examples that lend credence to the claim.
//   - This may be a reference to a text chunk from which the claim was extracted from.
//   - Or it may be a reference to other claims.

// imagine a human:
// look through all the headings, see which one it probably belongs to
// look through sub headings, determine if one needs to be added
// add it to one?
// occassionally restructure everything

// Creating concepts (or abstraction):
// - conceptualizing/grouping claims together is done when there are too many claims to make sense of.
// - conceptualizing allows one to use one "general" claim to encompass many specific claims.
// - creating concepts doesn't happen instantaneously, it is also method/process based.
//   - when you have a large set of possibly unrelated claims, you need to go through it systematically.
//   - you start with one claim, look at the groups/unassigned claims you have, see if it fits in any:
//		- if it follows an existing concept, put it in an existing concept
//      - if with an unassigned claim, make a new group with the two together
//   	- if not, put it in the unassigned bucket
//   -

// Organizer
// Operations:
//
// - list_groups(): (id, title, claims)[] -> list groups
// - list_claims(): (id, text) -> list the claims in a specific group, or that have not been assigned to a group
//
// - new_group(title, claims) -> group claims together into a new group
// - add_to_group(id, claim) -> add a claim to an existing group
// - remove_from_group(id, claim) -> remove a claim from an existing group

const prompt = `You are a research orchestrator who takes claims from research assistants and
You will use a multi-step process to understand the entire text.
You can call the "next_text" tool to

`

func run_cmp() (err error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	err = client.Chat(ctx, &api.ChatRequest{
		Model: gen_model,
		Messages: []api.Message{
			{
				Role:    "system",
				Content: fmt.Sprintf(prompt, docs),
			},
		},
	}, func(cr api.ChatResponse) error {
		fmt.Print(cr.Message.Content)
		return nil
	})
	if err != nil {
		return
	}

	return
}

func run() (err error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	// webui found at http://127.0.0.1:9091/webui
	vdb, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: "127.0.0.1:19530",
	})
	if err != nil {
		return
	}
	defer vdb.Close(ctx)

	err = IndexFiles(ctx, client, vdb, true)
	if err != nil {
		return
	}
	err = vdb.UseDatabase(ctx, milvusclient.NewUseDatabaseOption("nvim"))
	if err != nil {
		return
	}

	fmt.Println("embedding query")
	resp, err := client.Embed(ctx, &api.EmbedRequest{
		Model: embed_model,
		Input: "disable spell check neovim",
	})
	if err != nil {
		return
	}

	fmt.Println("loading collection")
	task, err := vdb.LoadCollection(ctx, milvusclient.NewLoadCollectionOption("docs"))
	if err != nil {
		return
	}
	err = task.Await(ctx)
	if err != nil {
		return
	}

	results, err := vdb.Search(ctx, milvusclient.NewSearchOption(
		"docs",
		10,
		[]entity.Vector{
			entity.FloatVector(resp.Embeddings[0]),
		},
	).WithANNSField("content"))
	if err != nil {
		return
	}
	for _, r := range results {
		ids := make([]int64, r.IDs.Len())
		for i := range r.IDs.Len() {
			ids[i], err = r.IDs.GetAsInt64(i)
			if err != nil {
				return
			}
		}

		fmt.Println("IDs:", ids)
		fmt.Println("Scores:", r.Scores)

		var entities milvusclient.ResultSet
		entities, err = vdb.Get(ctx, milvusclient.NewQueryOption("docs").
			WithConsistencyLevel(entity.ClStrong).
			WithIDs(column.NewColumnInt64("id", ids)).
			WithOutputFields("filename", "line_start", "line_end"))
		if err != nil {
			return
		}

		for i := range entities.Fields.Len() {
			for _, field := range entities.Fields {
				var v any
				v, err = field.Get(i)
				if err != nil {
					return
				}
				fmt.Print(v, " ")
			}
			fmt.Println()
		}
	}

	err = vdb.ReleaseCollection(ctx, milvusclient.NewReleaseCollectionOption("docs"))
	if err != nil {
		return
	}

	// client.Chat(ctx, &api.ChatRequest{
	// 	Model: model,
	// })
	return
}
