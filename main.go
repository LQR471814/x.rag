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

const prompt = `You are a researcher taking comprehensive notes on existing documentation.
You will use a multi-step process to understand the entire text.

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
