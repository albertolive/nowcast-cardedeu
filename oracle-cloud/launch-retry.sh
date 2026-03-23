#!/bin/bash
# =============================================================================
# Oracle Cloud Free Tier — Auto-retry instance launch
# Tries A1.Flex (ARM) first, then E2.1.Micro (AMD) every 60s until success
# =============================================================================
set -euo pipefail
export SUPPRESS_LABEL_WARNING=True

COMPARTMENT="ocid1.tenancy.oc1..aaaaaaaaqly6jh6rj6inx4lvqy4pi2zboqytbxrchddfzh5qfd5fojqyhovq"
AD="Hgty:EU-MADRID-1-AD-1"
SUBNET="ocid1.subnet.oc1.eu-madrid-1.aaaaaaaa35bqtdbmlkwyl6azdrfbynuz2pw5jxgv23uaz2xd3kcstbfdcs2a"
IMAGE_ARM="ocid1.image.oc1.eu-madrid-1.aaaaaaaaexzqv7p75wol73w6lss4f3odwhzxcntzaivzge4ywvhb4clowlba"
SSH_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILeFMupW1iwptZvziMcfFR9p/3z9Um8HiE6AtDyA2UsJ albertolivecorbella@gmail.com"

ATTEMPT=0
MAX_ATTEMPTS=720  # 12 hours at 60s intervals

echo "🔄 Starting Oracle Cloud instance launch retry loop..."
echo "   Will try A1.Flex (ARM 1 OCPU/6GB) and E2.1.Micro (AMD 1 OCPU/1GB)"
echo "   Retrying every 60 seconds. Press Ctrl+C to stop."
echo ""

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    TIMESTAMP=$(date '+%H:%M:%S')

    # ── Try ARM A1.Flex first (better specs) ──
    echo "[$TIMESTAMP] Attempt $ATTEMPT — Trying VM.Standard.A1.Flex (ARM)..."
    RESULT=$(oci compute instance launch \
        --compartment-id "$COMPARTMENT" \
        --availability-domain "$AD" \
        --shape "VM.Standard.A1.Flex" \
        --shape-config '{"ocpus": 1, "memoryInGBs": 6}' \
        --image-id "$IMAGE_ARM" \
        --subnet-id "$SUBNET" \
        --assign-public-ip true \
        --display-name "nowcast-cardedeu" \
        --ssh-authorized-keys-file /dev/stdin \
        --metadata '{"user_data": ""}' \
        2>&1 <<< "$SSH_KEY") || true

    if echo "$RESULT" | grep -q '"lifecycle-state"'; then
        echo ""
        echo "✅ SUCCESS! ARM instance created!"
        echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  ID: {d[\"data\"][\"id\"]}')" 2>/dev/null || true
        echo "  Waiting for public IP..."
        sleep 30
        INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['id'])" 2>/dev/null)
        if [ -n "$INSTANCE_ID" ]; then
            oci compute instance list-vnics --instance-id "$INSTANCE_ID" --query 'data[0]."public-ip"' --raw-output 2>/dev/null | grep -v SyntaxWarning || echo "  (check OCI console for public IP)"
        fi
        exit 0
    fi

    if ! echo "$RESULT" | grep -qi "out of.*capacity\|InternalError\|LimitExceeded"; then
        echo "  ⚠️  Unexpected error on ARM:"
        echo "$RESULT" | head -5
    fi

    # ── Try AMD E2.1.Micro as fallback ──
    echo "[$TIMESTAMP] Attempt $ATTEMPT — Trying VM.Standard.E2.1.Micro (AMD)..."

    # Get AMD image
    IMAGE_AMD=$(oci compute image list \
        --compartment-id "$COMPARTMENT" \
        --operating-system "Oracle Linux" \
        --operating-system-version "9" \
        --shape "VM.Standard.E2.1.Micro" \
        --query 'data[0].id' --raw-output 2>/dev/null | grep -v SyntaxWarning) || true

    if [ -n "$IMAGE_AMD" ]; then
        RESULT=$(oci compute instance launch \
            --compartment-id "$COMPARTMENT" \
            --availability-domain "$AD" \
            --shape "VM.Standard.E2.1.Micro" \
            --image-id "$IMAGE_AMD" \
            --subnet-id "$SUBNET" \
            --assign-public-ip true \
            --display-name "nowcast-cardedeu" \
            --ssh-authorized-keys-file /dev/stdin \
            --metadata '{"user_data": ""}' \
            2>&1 <<< "$SSH_KEY") || true

        if echo "$RESULT" | grep -q '"lifecycle-state"'; then
            echo ""
            echo "✅ SUCCESS! AMD Micro instance created!"
            echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  ID: {d[\"data\"][\"id\"]}')" 2>/dev/null || true
            echo "  Waiting for public IP..."
            sleep 30
            INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['id'])" 2>/dev/null)
            if [ -n "$INSTANCE_ID" ]; then
                oci compute instance list-vnics --instance-id "$INSTANCE_ID" --query 'data[0]."public-ip"' --raw-output 2>/dev/null | grep -v SyntaxWarning || echo "  (check OCI console for public IP)"
            fi
            exit 0
        fi

        if ! echo "$RESULT" | grep -qi "out of.*capacity\|InternalError\|LimitExceeded"; then
            echo "  ⚠️  Unexpected error on AMD:"
            echo "$RESULT" | head -5
        fi
    fi

    echo "  Both shapes out of capacity. Retrying in 60s..."
    sleep 60
done

echo "❌ Max attempts reached ($MAX_ATTEMPTS). Try again later or try a different region."
exit 1
