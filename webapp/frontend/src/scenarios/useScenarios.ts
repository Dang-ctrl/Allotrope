import { useEffect, useState } from 'react'
import type { Scenarios } from './types'

type State =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: Scenarios }

/** Fetches the static scenarios.json (public/scenarios.json), written by
 * scripts/generate_scenarios.py. Regenerate it and re-copy it into
 * public/ after changing anything upstream of these scenarios -- see
 * context.md's "Judge-facing artifacts" section. */
export function useScenarios(): State {
  const [state, setState] = useState<State>({ status: 'loading' })

  useEffect(() => {
    let cancelled = false
    fetch('/scenarios.json')
      .then((res) => {
        if (!res.ok) throw new Error(`GET /scenarios.json -> ${res.status}`)
        return res.json()
      })
      .then((data: Scenarios) => {
        if (!cancelled) setState({ status: 'ready', data })
      })
      .catch((err: Error) => {
        if (!cancelled) setState({ status: 'error', message: err.message })
      })
    return () => {
      cancelled = true
    }
  }, [])

  return state
}
