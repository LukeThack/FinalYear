from esa_snappy import ProductIO, GPF, HashMap, jpy
import os
File = jpy.get_type("java.io.File")
BandDescriptor = jpy.get_type(
    "org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor")
Integer = jpy.get_type('java.lang.Integer')

# input_file="satelite/S1A_IW_GRDH_1SDV_20230612T063111_20230612T063136_048949_05E2DC_F290.zip"
# output_file="satelite/image11.dim"

'''
Takes a sentinel 1 zip file and applys several filters and mask to the image then saves the result.

Parameters:
    input_file(string): path and name of the input dim file
    output_file(string): path and name of the dim output
'''


def process_SAR_image(input_file, output_file):
    product = ProductIO.readProduct(input_file)
    orbit_params = HashMap()
    orbit_params.put("Apply-Orbit-File", True)
    orbit_params.put("Orbit_Type", "Sentinel Precise (Auto Download)")
    # Improves geolocation accuracy
    orbit_product = GPF.createProduct(
        "Apply-Orbit-File", orbit_params, product)

    calibration_params = HashMap()
    calibration_params.put("outputSigmaBand", False)
    calibration_params.put("outputGammaBand", True)
    calibration_params.put("selectedPolarisation", "VV")
    calibration_params.put("sourceBands", "Intensity_VV")
    # Applies radiometric calibration, converting to Gamma0 backscatter (reflectivity corrected for terrain slope)
    calibrated_product = GPF.createProduct(
        "Calibration", calibration_params, orbit_product)

    terrain_correction_params = HashMap()
    terrain_correction_params.put("demName", "SRTM 1Sec HGT")
    terrain_correction_params.put("imgResamplingMethod", "NEAREST_NEIGHBOUR")
    terrain_correction_params.put("pixelSpacingInMeter", 10.0)
    terrain_correction_params.put("mapProjection", "WGS84(DD)")
    terrain_correction_params.put("saveSelectedSourceBand", True)
    terrain_correction_params.put("nodataValueAtSea", False)
    # Uses a digital elevation model to correct SAR image
    terrain_product = GPF.createProduct(
        "Terrain-Correction", terrain_correction_params, calibrated_product)

    vector_params = HashMap()
    vector_params.put("vectorFile", "Mask/UKIRMASK.shp")
    vector_params.put("separateShapes", False)  # make shp file into one mask
    vector_params.put("createMasks", True)
    product_with_vector = GPF.createProduct(
        "Import-Vector", vector_params, terrain_product)

    BandDescriptor = jpy.get_type(
        "org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor")
    band = BandDescriptor()
    band.name = "Gamma0_VV_ocean"
    band.type = "float32"
    band.unit = "linear"
    # Applies mask, set anything that is within the mask to Nan
    band.expression = "if (UKIRMASK==0) then Gamma0_VV else NaN"
    band_array = jpy.array(BandDescriptor, 1)
    band_array[0] = band
    bm_params = HashMap()
    bm_params.put("targetBandDescriptors", band_array)
    masked_product = GPF.createProduct(
        "BandMaths", bm_params, product_with_vector)

    ProductIO.writeProduct(masked_product, output_file, "BEAM-DIMAP")


'''
Processes every .dim file in a directory through the process_SAR_image function.

Parameters:
    directory(string): path to directory to process
'''


def process_directory(directory):
    all_files = os.listdir(directory)
    final_image_id = next_id(all_files, ".dim")
    for file in all_files:
        if file.endswith(".zip"):
            input_file = os.path.join(directory, file)
            output_file = os.path.join(
                directory, "image{}.dim".format(final_image_id))
            process_SAR_image(input_file, output_file)
            final_image_id += 1


'''
Finds the largest number found in a directory (only if there is one number in the file name)

Parameters:
    all_files(string list): list of names of files
    file_extension(string): the file extension of the files user wants to read.

Returns:
    int: the largest number found in that directory
'''


def next_id(all_files, file_extension=""):
    final_image_ids = []
    for file in all_files:
        if file.endswith(file_extension):
            name = os.path.basename(file)
            name_list = list(name)
            num = ""
            for char in name_list:
                if char.isdigit():
                    num += char
            final_image_id = int(num)+1
            final_image_ids.append(final_image_id)

    final_image_id = max(final_image_ids) if final_image_ids else 0
    return final_image_id
