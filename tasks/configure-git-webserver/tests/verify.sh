#!/bin/bash
# Test the git-to-web deployment without allowing broken SSH setups to block
# the verifier until Harbor's outer timeout.

set -u

fail() {
    echo "❌ TEST FAILED: $1"
    echo "Test completed"
    exit 1
}

for required_command in git ssh ssh-keygen curl timeout; do
    command -v "$required_command" >/dev/null 2>&1 \
        || fail "Required command is unavailable: $required_command"
done

id user >/dev/null 2>&1 || fail "Required SSH account 'user' does not exist"

USER_HOME=$(getent passwd user | cut -d: -f6)
USER_GROUP=$(id -gn user)
[ -n "$USER_HOME" ] || fail "Could not determine the home directory for 'user'"

VERIFY_TMP=$(mktemp -d) || fail "Could not create verifier workspace"
trap 'rm -rf "$VERIFY_TMP"' EXIT

# Set up the login credentials promised by the task. The evaluated setup must
# still provide the account, SSH server, repository, hook, and web server.
ssh-keygen -t ed25519 -f "$VERIFY_TMP/id_ed25519" -N "" -q \
    || fail "Could not generate verifier SSH key"
install -d -m 700 -o user -g "$USER_GROUP" "$USER_HOME/.ssh" \
    || fail "Could not configure SSH directory for 'user'"
install -m 600 -o user -g "$USER_GROUP" \
    "$VERIFY_TMP/id_ed25519.pub" "$USER_HOME/.ssh/authorized_keys" \
    || fail "Could not authorize verifier SSH key"

export GIT_SSH_COMMAND="ssh -i $VERIFY_TMP/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

if ! timeout --kill-after=5s 30s \
    git clone user@localhost:/git/server "$VERIFY_TMP/server-test"; then
    fail "Could not clone the repository over SSH"
fi

echo 'hello world' > "$VERIFY_TMP/server-test/hello.html"
git -C "$VERIFY_TMP/server-test" add hello.html \
    || fail "Could not stage hello.html"
git -C "$VERIFY_TMP/server-test" config user.email 'test@example.com'
git -C "$VERIFY_TMP/server-test" config user.name 'Test User'
git -C "$VERIFY_TMP/server-test" commit --allow-empty -m 'Add hello.html' \
    || fail "Could not commit hello.html"
if ! timeout --kill-after=5s 30s \
    git -C "$VERIFY_TMP/server-test" push origin master; then
    fail "Could not push to the repository over SSH"
fi

echo "Testing web server..."
sleep 2

echo "Using curl to test web server..."
if ! HTTP_RESPONSE=$(curl --silent --output /dev/null --write-out "%{http_code}" \
    --connect-timeout 5 --max-time 10 http://localhost:8080/hello.html); then
    fail "Could not connect to the web server"
fi

if [ "$HTTP_RESPONSE" != "200" ]; then
    fail "Web server returned HTTP $HTTP_RESPONSE"
fi

if ! CONTENT=$(curl --silent --show-error --connect-timeout 5 --max-time 10 \
    http://localhost:8080/hello.html); then
    fail "Could not retrieve hello.html from the web server"
fi

echo "Web server test successful (HTTP 200)"
echo "Content: $CONTENT"
if [ "$CONTENT" != "hello world" ]; then
    echo "Expected: 'hello world'"
    echo "Got: '$CONTENT'"
    fail "Content does not match expected output"
fi

echo "✅ TEST PASSED: Content matches expected output"
echo "Test completed"
