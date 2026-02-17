from esa_snappy import ProductIO, GPF,HashMap,jpy
File = jpy.get_type("java.io.File")
BandDescriptor = jpy.get_type("org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor")
Integer = jpy.get_type('java.lang.Integer')

input_file="satelite/S1A_IW_GRDH_1SDV_20230603T180703_20230603T180728_048825_05DF24_0616.zip"
output_file="satelite/image3.dim"


def process_SAR_image(input_file,output_file):
    product=ProductIO.readProduct(input_file)
    orbit_params=HashMap()
    orbit_params.put("Apply-Orbit-File",True)
    orbit_params.put("Orbit_Type","Sentinel Precise (Auto Download)")
    orbit_product=GPF.createProduct("Apply-Orbit-File",orbit_params,product)

    calibration_params=HashMap()
    calibration_params.put("outputSigmaBand", False)
    calibration_params.put("outputGammaBand", True)
    calibration_params.put("selectedPolarisation","VV")
    calibration_params.put("sourceBands","Intensity_VV")
    calibrated_product=GPF.createProduct("Calibration",calibration_params,orbit_product)

    terrain_correction_params=HashMap()
    terrain_correction_params.put("demName","SRTM 1Sec HGT")
    terrain_correction_params.put("imgResamplingMethod","NEAREST_NEIGHBOUR")
    terrain_correction_params.put("pixelSpacingInMeter",10.0)
    terrain_correction_params.put("mapProjection","WGS84(DD)")
    terrain_correction_params.put("saveSelectedSourceBand",True)
    terrain_correction_params.put("nodataValueAtSea",False)
    terrain_product=GPF.createProduct("Terrain-Correction",terrain_correction_params,calibrated_product)

    vector_params=HashMap()
    vector_params.put("vectorFile","Mask/UKIRMASK.shp")
    vector_params.put("separateShapes",False) #make shp file into one mask
    vector_params.put("createMasks",True)
    product_with_vector = GPF.createProduct("Import-Vector",vector_params,terrain_product)

    BandDescriptor = jpy.get_type("org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor")
    band=BandDescriptor()
    band.name="Gamma0_VV_ocean"     
    band.type="float32"    
    band.unit = "linear"          
    band.expression="if (UKIRMASK==0) then Gamma0_VV else NaN"
    band_array=jpy.array(BandDescriptor,1)
    band_array[0]=band
    bm_params=HashMap()
    bm_params.put("targetBandDescriptors",band_array)
    masked_product=GPF.createProduct("BandMaths",bm_params,product_with_vector)

    ProductIO.writeProduct(masked_product,output_file,"BEAM-DIMAP")
process_SAR_image(input_file,output_file)