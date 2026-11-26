# Frequently asked questions

## `bwrap: setting up uid map: Permission denied` on Ubuntu 23.10+/24.04

Ubuntu restricts unprivileged user namespaces through AppArmor
(`kernel.apparmor_restrict_unprivileged_userns=1`).  The `bwrap` from the pixi
environment — `<repo>/.pixi/envs/default/bin/bwrap`, which the fleet scripts put
first on `PATH` — has no AppArmor profile, so the kernel refuses bwrap's uid map
and `scripts/fleet/run_*_h4l.sh` exits before the agent starts.  This is a host
policy, not a repository bug.

Give that one binary the `userns` permission with a scoped profile.  From the
repository root:

```bash
BWRAP="$(readlink -f .pixi/envs/default/bin/bwrap)"
sudo tee /etc/apparmor.d/hep-ex-bench-bwrap >/dev/null <<EOF
abi <abi/4.0>,
include <tunables/global>

profile hep-ex-bench-bwrap $BWRAP flags=(unconfined) {
  userns,
  include if exists <local/hep-ex-bench-bwrap>
}
EOF
sudo apparmor_parser -r /etc/apparmor.d/hep-ex-bench-bwrap
```

`flags=(unconfined)` leaves bwrap otherwise as unrestricted as it is today;
`userns` is the only permission the profile adds, and only to that executable.
It applies to new processes immediately and is loaded again at boot from
`/etc/apparmor.d/`.  AppArmor 4.0+ (Ubuntu 24.04+) is required for the `userns`
rule.  Re-run the commands if the checkout moves.  If you also need
`/usr/bin/bwrap`, add a second profile with its own name and that one attachment
path; a profile line accepts only one path.

Verify:

```bash
pixi run sandbox    # bwrap: True
d=$(mktemp -d); scripts/fleet/dev_sandbox.sh "$d" -- /bin/true && echo "dev sandbox: ok"; rm -rf "$d"
```

`unshare -rn` can remain `False`: only bwrap is exempted, and the runner prefers
bwrap.

Undo (unload first, then remove the file so it does not return at boot):

```bash
sudo apparmor_parser -R /etc/apparmor.d/hep-ex-bench-bwrap
sudo rm /etc/apparmor.d/hep-ex-bench-bwrap
```

`sudo aa-status | grep hep-ex-bench` should then print nothing.  This is
preferable to `kernel.apparmor_restrict_unprivileged_userns=0`, which relaxes
the restriction for every process on the host.  Note that the pixi environment
is user-writable: the profile trusts anything that can modify that bwrap path.
