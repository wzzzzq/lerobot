cd assets
python _download.py

cd /pfs/pfs-ilWc5D/ziqianwang/robotwin

# background_texture
unzip background_texture.zip
rm -rf background_texture.zip

# embodiments
unzip embodiments.zip
rm -rf embodiments.zip

# objects
unzip objects.zip
rm -rf objects.zip

cd ..
cp -r assets/files /pfs/pfs-ilWc5D/ziqianwang/robotwin/
rm -rf assets
ln -s /pfs/pfs-ilWc5D/ziqianwang/robotwin ./assets
echo "Configuring Path ..."
python ./script/update_embodiment_config_path.py

huggingface-cli download TianxingChen/RoboTwin2.0 \
  --repo-type dataset \
  --local-dir $RAW_DATA_DIR\
  --include "dataset/aloha-agilex/stack_bowls_.zip" \
  --resume-download \
  --max-workers 1  