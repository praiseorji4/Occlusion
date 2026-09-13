#!/usr/bin/env bash
# Clear the "Could not resolve file [hatchback.png]" style errors from Gazebo.
#
# Several Fuel vehicle models ship material files that reference their textures in ways
# Gazebo cannot resolve from the Fuel cache:
#
#   1. by BARE FILENAME            e.g. "ambulance.png"
#      Gazebo looks next to the mesh, but Fuel stores textures in materials/textures/.
#
#   2. by SHORT MODEL URI          e.g. "model://suv/materials/textures/wheels_01.png"
#      There is no model called plain "suv" on the resource path - the cache directory is
#      .../openrobotics/models/suv/<version>, and the hatchback is even cached under the
#      literal name "hatchback copy".
#
# This is a packaging problem in those models, not in this repo. The fix is local and
# safe: symlink the textures next to the meshes, and build a directory of short-name
# aliases to add to GZ_SIM_RESOURCE_PATH. Re-run it any time; it is idempotent.
#
#   bash scripts/fix_fuel_textures.sh
#
set -u

FUEL="${HOME}/.gz/fuel/fuel.gazebosim.org"
ALIAS_DIR="${HOME}/.gz/fuel_aliases"

if [ ! -d "$FUEL" ]; then
  echo "No Fuel cache at $FUEL - run a world once so Gazebo downloads the models, then re-run."
  exit 1
fi

echo "1. linking textures next to meshes (fixes bare-filename references)"
linked=0
while IFS= read -r -d '' texdir; do
  model_ver_dir="$(dirname "$(dirname "$texdir")")"
  meshdir="$model_ver_dir/meshes"
  [ -d "$meshdir" ] || continue
  for tex in "$texdir"/*; do
    [ -f "$tex" ] || continue
    target="$meshdir/$(basename "$tex")"
    if [ ! -e "$target" ]; then
      ln -s "$tex" "$target" && linked=$((linked + 1))
    fi
  done
done < <(find "$FUEL" -type d -path "*/materials/textures" -print0)
echo "   $linked texture links created"

echo "2. creating short-name aliases for model:// references"
mkdir -p "$ALIAS_DIR"
aliased=0
while IFS= read -r -d '' modeldir; do
  name="$(basename "$modeldir")"
  # newest version directory of this model
  ver="$(ls "$modeldir" 2>/dev/null | sort -n | tail -1)"
  [ -n "$ver" ] || continue
  short="$(echo "$name" | sed 's/ copy$//; s/ /_/g')"
  for candidate in "$short" "$(echo "$short" | tr '_' ' ')"; do
    target="$ALIAS_DIR/$candidate"
    if [ ! -e "$target" ]; then
      ln -s "$modeldir/$ver" "$target" 2>/dev/null && aliased=$((aliased + 1))
    fi
  done
done < <(find "$FUEL" -mindepth 3 -maxdepth 3 -type d -path "*/models/*" -print0)
echo "   $aliased aliases in $ALIAS_DIR"

echo
echo "Add the alias directory to Gazebo's search path (put this in ~/.bashrc to keep it):"
echo
echo "    export GZ_SIM_RESOURCE_PATH=\$GZ_SIM_RESOURCE_PATH:$ALIAS_DIR"
echo
echo "Then relaunch. Remaining texture warnings are cosmetic - they only affect how the"
echo "borrowed vehicle models look, never the geometry, the sensors or the physics."
