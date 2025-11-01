DATA_DIR="/data"
cd $DATA_DIR
wget --user-agent="Mozilla/5.0" \
     -c "https://ucla.app.box.com/index.php?rm=box_download_shared_file&vanity_name=UCLA-MobilityLab-OPV2V&file_id=f_1621828681754" \
     -O OPV2V.zip
unzip OPV2V.zip
rm OPV2V.zip
cd -
