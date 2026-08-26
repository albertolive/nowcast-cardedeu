# Watchdog heartbeat update — apply AFTER the VM is live

`watchdog.yml`'s "Check dispatch heartbeat" step counts `workflow_dispatch`
runs as proof the trigger side works. Once the CF Worker stops dispatching,
that count is always 0 and the watchdog false-alarms every hour.

The same guarantee (a dead tick-side leaves no trace that the freshness
check could miss) now comes from the container's own push-back commits:
every successful tick commits `data/latest_prediction.json`. Count those.

Replace the whole `- name: Check dispatch heartbeat` step with:

```yaml
      - name: Check prediction-commit heartbeat
        id: heartbeat
        # The OCI VM runs predict in-process and pushes one commit per tick
        # (~8 per 80-min window). A missing tick leaves NO trace except the
        # absence of these commits, and the freshness check above masks a
        # slow leak (one or two surviving ticks keep the output fresh). So
        # count commits touching the state file directly; too few means the
        # runtime or its git push-back is broken (VM stopped, GIT_TOKEN
        # expired) even while predictions still look fresh.
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          SINCE=$(date -u -d '80 minutes ago' +%Y-%m-%dT%H:%M:%SZ)
          COUNT=$(gh api "repos/${GITHUB_REPOSITORY}/commits?sha=main&path=data/latest_prediction.json&since=${SINCE}&per_page=100" --jq 'length')
          echo "Prediction commits since ${SINCE}: ${COUNT}"
          if [ "${COUNT}" -lt 6 ]; then
            echo "gap=true" >> "$GITHUB_OUTPUT"
            echo "count=${COUNT}" >> "$GITHUB_OUTPUT"
            exit 1
          fi
```

Everything else in watchdog.yml (freshness threshold, radar blindness,
Telegram alerting, hourly schedule) stays exactly as it is — the
cross-platform backstop design is unchanged: Actions watches the VM from
the outside.
