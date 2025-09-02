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
	client, err := api.ClientFromEnvironment()
	if err != nil {
		panic(err)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	// webui found at http://127.0.0.1:9091/webui
	vdb, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: "127.0.0.1:19530",
	})
	if err != nil {
		panic(err)
	}
	defer vdb.Close(ctx)

	err = IndexFiles(ctx, client, vdb, false)
	if err != nil {
		panic(err)
	}
	err = vdb.UseDatabase(ctx, milvusclient.NewUseDatabaseOption("nvim"))
	if err != nil {
		panic(err)
	}

	fmt.Println("embedding query")
	resp, err := client.Embed(ctx, &api.EmbedRequest{
		Model: embed_model,
		Input: "disable spell check neovim",
	})
	if err != nil {
		panic(err)
	}

	fmt.Println("loading collection")
	task, err := vdb.LoadCollection(ctx, milvusclient.NewLoadCollectionOption("docs"))
	if err != nil {
		panic(err)
	}
	err = task.Await(ctx)
	if err != nil {
		panic(err)
	}

	results, err := vdb.Search(ctx, milvusclient.NewSearchOption(
		"docs",
		10,
		[]entity.Vector{
			entity.FloatVector(resp.Embeddings[0]),
		},
	).WithANNSField("content"))
	if err != nil {
		panic(err)
	}
	for _, r := range results {
		ids := make([]int64, r.IDs.Len())
		for i := range r.IDs.Len() {
			ids[i], err = r.IDs.GetAsInt64(i)
			if err != nil {
				panic(err)
			}
		}

		fmt.Println("IDs:", ids)
		fmt.Println("Scores:", r.Scores)

		entities, err := vdb.Get(ctx, milvusclient.NewQueryOption("docs").
			WithConsistencyLevel(entity.ClStrong).
			WithIDs(column.NewColumnInt64("id", ids)).
			WithOutputFields("filename", "line_start", "line_end"))
		if err != nil {
			panic(err)
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
		panic(err)
	}

	// client.Chat(ctx, &api.ChatRequest{
	// 	Model: model,
	// })
}
