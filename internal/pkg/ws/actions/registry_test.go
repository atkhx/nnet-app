package actions

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRegistry_Add(t *testing.T) {
	t.Run("AddNew", func(t *testing.T) {
		callsCount := 0
		actionName := "name1"
		actionFunc := func(ctx context.Context) error {
			callsCount++
			return nil
		}

		actions := NewRegistry()
		actions.Add(actionName, actionFunc)

		actualActionFunc, actualErr := actions.Get(actionName)

		if assert.NoError(t, actualErr) {
			assert.NoError(t, actualActionFunc(context.Background()))
			assert.Equal(t, 1, callsCount)
		}
	})

	t.Run("OverrideExists", func(t *testing.T) {
		callsCount := 0
		actionName := "name1"

		actionFunc1 := func(ctx context.Context) error {
			assert.Fail(t, "should not be executed")
			return nil
		}

		actionFunc2 := func(ctx context.Context) error {
			callsCount++
			return nil
		}

		actions := NewRegistry()
		actions.Add(actionName, actionFunc1)
		actions.Add(actionName, actionFunc2)

		actualActionFunc, actualErr := actions.Get(actionName)

		if assert.NoError(t, actualErr) {
			assert.NoError(t, actualActionFunc(context.Background()))
			assert.Equal(t, 1, callsCount)
		}
	})
}

func TestRegistry_Get(t *testing.T) {
	t.Run("OnEmpty", func(t *testing.T) {
		actions := NewRegistry()
		actualActionFunc, actualErr := actions.Get("unknown")

		assert.Nil(t, actualActionFunc)
		assert.Error(t, actualErr)
		assert.Equal(t, ErrActionNotFound, actualErr)
	})

	t.Run("Unknown", func(t *testing.T) {
		actions := NewRegistry()
		actions.Add("name1", func(ctx context.Context) error {
			return errors.New("func1 should not be executed")
		})

		actualActionFunc, actualErr := actions.Get("name2")

		assert.Nil(t, actualActionFunc)
		assert.Error(t, actualErr)
		assert.Equal(t, ErrActionNotFound, actualErr)
	})

	t.Run("Exists", func(t *testing.T) {
		callsCount := 0

		actions := NewRegistry()
		actions.Add("name1", func(ctx context.Context) error {
			return errors.New("func1 should not be executed")
		})

		actions.Add("name2", func(ctx context.Context) error {
			callsCount++
			return nil
		})

		actions.Add("name3", func(ctx context.Context) error {
			return errors.New("func3 should not be executed")
		})

		actualActionFunc, actualErr := actions.Get("name2")
		if assert.NoError(t, actualErr) {
			assert.NoError(t, actualActionFunc(context.Background()))
			assert.Equal(t, 1, callsCount)
		}
	})
}

func TestRegistry_Del(t *testing.T) {
	t.Run("OnEmpty", func(t *testing.T) {
		actions := NewRegistry()
		actions.Del("name1")
	})

	t.Run("Unknown", func(t *testing.T) {
		actions := NewRegistry()
		actions.Add("name1", func(ctx context.Context) error {
			return errors.New("func1 should not be executed")
		})
		actions.Del("name2")

		actualActionFunc, actualErr := actions.Get("name1")
		assert.NotNil(t, actualActionFunc)
		assert.NoError(t, actualErr)
	})

	t.Run("Exists", func(t *testing.T) {
		actions := NewRegistry()
		actions.Add("name1", func(ctx context.Context) error {
			return errors.New("func1 should not be executed")
		})
		actions.Del("name1")

		actualActionFunc, actualErr := actions.Get("name1")

		assert.Nil(t, actualActionFunc)
		assert.Error(t, actualErr)
		assert.Equal(t, ErrActionNotFound, actualErr)
	})
}
