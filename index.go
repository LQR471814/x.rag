package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/ollama/ollama/api"
)

// chunk size in # of chunks
const chunk_size = 5

// chunk margin in # of lines
const chunk_margin = 3

func retrieveMarginEnd(lines []string, margin int) []string {
	if len(lines) == 0 {
		return nil
	}
	i := max(len(lines)-margin, 0)
	return lines[i:]
}

func readChunks(f *os.File, onChunk func(ls, le int, s string)) (err error) {
	var accumulator []string
	var prevChunk []string

	pcount := 0

	linepos := 0
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if line != "" {
			accumulator = append(accumulator, line)
			linepos++
			continue
		}
		pcount++
		if pcount >= chunk_size && len(accumulator) > 0 {
			pcount = 0
			out := append(retrieveMarginEnd(prevChunk, chunk_margin), accumulator...)
			onChunk(min(0, linepos-len(out)), linepos, strings.Join(out, "\n"))
			prevChunk = accumulator
			accumulator = nil
		}
	}

	if len(accumulator) > 0 {
		out := append(retrieveMarginEnd(prevChunk, chunk_margin), accumulator...)
		onChunk(linepos-len(out), linepos, strings.Join(out, "\n"))
	}

	return
}

type chunk struct {
	filename   string
	content    string
	line_start int
	line_end   int
}

func readDocs(chunks chan chunk) (err error) {
	entries, err := os.ReadDir("nvim-docs")
	if err != nil {
		return
	}
	defer close(chunks)
	for _, e := range entries {
		var f *os.File
		f, err = os.Open(filepath.Join("nvim-docs", e.Name()))
		if err != nil {
			return
		}
		err = readChunks(f, func(ls, le int, s string) {
			chunks <- chunk{
				filename:   e.Name(),
				content:    s,
				line_start: ls,
				line_end:   le,
			}
		})
		if err != nil {
			return
		}
	}
	return
}

func indexWorker(ctx context.Context, wg *sync.WaitGroup, client *api.Client, vdb *milvusclient.Client, chunks chan chunk) (err error) {
	defer wg.Done()
	for c := range chunks {
		var res *api.EmbedResponse
		res, err = client.Embed(ctx, &api.EmbedRequest{
			Model: embed_model,
			Input: c.content,
		})
		if err != nil {
			return
		}
		_, err = vdb.Insert(ctx, milvusclient.
			NewColumnBasedInsertOption("docs").
			WithInt64Column("line_start", []int64{int64(c.line_start)}).
			WithInt64Column("line_end", []int64{int64(c.line_end)}).
			WithVarcharColumn("filename", []string{c.filename}).
			WithFloatVectorColumn("content", embed_dim, res.Embeddings))
		if err != nil {
			return
		}
		fmt.Println("insert chunk", c.filename)
	}
	return
}

func IndexFiles(ctx context.Context, client *api.Client, vdb *milvusclient.Client, overwrite bool) (err error) {
	databases, err := vdb.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil {
		return
	}
	if !slices.Contains(databases, "nvim") {
		err = vdb.CreateDatabase(ctx, milvusclient.NewCreateDatabaseOption("nvim"))
		if err != nil {
			return
		}
	} else if overwrite {
		err = vdb.UseDatabase(ctx, milvusclient.NewUseDatabaseOption("nvim"))
		if err != nil {
			return
		}
		var names []string
		names, err = vdb.ListCollections(ctx, milvusclient.NewListCollectionOption())
		if err != nil {
			return
		}
		for _, n := range names {
			err = vdb.DropCollection(ctx, milvusclient.NewDropCollectionOption(n))
			if err != nil {
				return
			}
		}

		err = vdb.DropDatabase(ctx, milvusclient.NewDropDatabaseOption("nvim"))
		if err != nil {
			return
		}
		err = vdb.CreateDatabase(ctx, milvusclient.NewCreateDatabaseOption("nvim"))
		if err != nil {
			return
		}
	} else {
		return
	}

	err = vdb.UseDatabase(ctx, milvusclient.NewUseDatabaseOption("nvim"))
	if err != nil {
		return
	}

	schema := entity.NewSchema().WithDynamicFieldEnabled(true).
		WithField(entity.NewField().WithName("id").WithIsAutoID(true).WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("filename").WithDataType(entity.FieldTypeVarChar).WithMaxLength(256)).
		WithField(entity.NewField().WithName("line_start").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("line_end").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("content").WithDataType(entity.FieldTypeFloatVector).WithDim(embed_dim))

	idx := index.NewAutoIndex(index.MetricType(entity.IP))
	err = vdb.CreateCollection(ctx, milvusclient.NewCreateCollectionOption("docs", schema).WithIndexOptions(
		milvusclient.NewCreateIndexOption("docs", "content", idx),
	))
	if err != nil {
		return
	}

	chunks := make(chan chunk, 4)
	go readDocs(chunks)

	wg := sync.WaitGroup{}
	for range runtime.NumCPU() {
		wg.Add(1)
		go indexWorker(ctx, &wg, client, vdb, chunks)
	}
	wg.Wait()

	return
}
