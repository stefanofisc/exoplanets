"""
    Author: Stefano Fiscale
    Date: 2025-07-04
    
    This file implements the class DataProcessor. This class defines the methods for generating global views
    starting from raw light curves.
"""
import  lightkurve      as lk
import  numpy           as np
import  yaml
import  csv
import  sys
from    scipy.signal    import  savgol_filter
from    pathlib         import  Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from    utils           import  GlobalPaths
from    columns_config  import  FittedEventsColumns

class DataProcessor:
    def __init__(self):
        """Constructor for the class DataProcessor"""
        self.__load_config(config_file = GlobalPaths.CONFIG / GlobalPaths.config_data_preparation)
    
    def __load_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.__sampling_rate        = config.get('sampling_rate', 30)
        self.__global_view_length   = config.get('global_view_length', 201)
        self.__fixed_depth          = config.get('fixed_depth', True)
    
    @property
    def sampling_rate(self):
        """
            Using @property to access "protected" attributes (__attribute) without making them public,
            thus maintaining encapsulation.
        """
        return self.__sampling_rate

    @property
    def global_view_length(self):
        return self.__global_view_length

    @property
    def fixed_depth(self):
        return self.__fixed_depth
    
    def _check_nan_in_flux(self, np_array):
        """
            Check for NaN in a numpy.ndarray

            Args:
                np_array (numpy.ndarray): The NumPy array to be checked.

            Returns:
                bool: True if the array contains at least one NaN, False otherwise.
        """
        # np.isnan(np_array) crea un array booleano dove True indica un NaN
        # np.any() controlla se c'è almeno un True nell'array booleano
        return np.any(np.isnan(np_array))

    def __nan_helper(self, y):
        """
            Helper to handle indices and logical indices of NaNs.

            Input:
                - y, 1d numpy array with possible NaNs
            Output:
                - nans, logical indices of NaNs
                - index, a function, with signature indices= index(logical_indices),
                to convert logical indices of NaNs to 'equivalent' indices
            Example:
                >>> # linear interpolation of NaNs
                >>> nans, x= self.__nan_helper(y)
                >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
            Source: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
        """

        return np.isnan(y), lambda z: z.nonzero()[0]

    def _interpolate_nan(self, global_view_flux):
        """
            Detect NaNs in the global view flux data and interpolate these values

            Input:
                - global_view_flux: a flux vector with possible NaNs in data
            Output:
                - global_view_flux: the same flux vector without NaN. Note that the length of the input vector is preserved.
        """
        nans, x= self.__nan_helper(global_view_flux)
        global_view_flux[nans]= np.interp(x(nans), x(~nans), global_view_flux[~nans])

        return global_view_flux
    
    def __zero_median_fixed_depth(self, global_view_flux):
        """
            To normalize the resulting views of each TCE, this method subtracts the median (np.median(global_view_flux)) from each view
            and then divide it by the absolute value of the minimum (np.min(global_view_flux)).

            Input:
                - global_view_flux: a flux vector;
            Output:
                - norm_global_view_flux: the normalized flux vector with median 0 and maximum transit depth -1
        """
        global_view_flux    -= np.median(global_view_flux, axis=0)

        min_val_abs         = np.abs(np.min(global_view_flux, axis=0))
        divisor             = np.where(min_val_abs == 0, 1e-10, min_val_abs)
        global_view_flux    /= divisor
        #global_view_flux /= np.abs(np.min(global_view_flux, axis=0))           #NOTE. This was the original code, but I added some controls to increase robustness against errors

        return global_view_flux

    def _zero_median(self, global_view_flux):
        """
            #NOTE. Implemento questo metodo perché in PLATO, per identificare i contaminant,
                    secondo me non devo normalizzare le profondità dei transiti.
            To normalize the resulting views of each TCE, this method subtracts 
            the median (np.median(global_view_flux)) from each view

            Input:
                - global_view_flux: a flux vector;
            Output:
                - norm_global_view_flux: the normalized flux vector with median 0
        """
        global_view_flux    -= np.median(global_view_flux, axis=0)

        return global_view_flux

    def lc_flattening(self, lc, transit_mask, window_length=11, polyorder=3):
        """
            Removes the low frequency trend and all the events that are not due to the event to be processed by using scipy’s Savitzky-Golay filter.
            Input:
            - lc_collection: a LightCurve object. The light curve to be flattened;
            - transit_mask: Mask that flags transits. Mask is True where there are transits of interest.
            - window_length: The length of the filter window (i.e. the number of coefficients). window_length must be a positive odd integer.
            - polyorder: The order of the polynomial used to fit the samples. polyorder must be less than window_length.
            Output:
            - the flattened LightCurve object. The transits of interest have been preserved during the flattening.
        """
        # Savitzky-Golay interpolation. The interpolated output signal has the same shape as the input
        my_savgol_filter = savgol_filter(lc.flux.value, window_length=window_length, polyorder=polyorder)
        
        # Flattening
        flattened_flux  = []
        for tm_i in range(len(transit_mask)):
            if transit_mask[tm_i] == False:
                flattened_flux.append( lc.flux.value[tm_i] / my_savgol_filter[tm_i] )
            else:
                # This is a transit point, do not flatten
                flattened_flux.append( lc.flux.value[tm_i] )
        
        # Return a LightCurve object containing the flattened flux data
        my_lc_flatten = lk.LightCurve(time=lc.time.value, flux=flattened_flux)

        return my_lc_flatten
    
    def lc_phase_folding(self, lc, t0, period, phase_range=1):
        """
            Phase fold a light curve over the input period.
            Compute the phase [0,1] (in cycles) for each observation (ti,mag_i), according to the following formula:
            phi = decimal part of [(ti-t0)/period]
            Input:
                - lc: lk.Lightcurve object
                - t0: epoch of the first transit
                - period: transit period (in days)
                - phase_range: value in [0, 1]. 1=fold over the entire phase
        """
        phi = []
        for i in range(len(lc.time.value)):
            phase = ( lc.time.value[i]-t0 ) / period
            # Remove the integer part because we are not interested in what cycle we are. We retain the decimal part only because we want to know in which part of the generic cycle we are.
            #phi.append( phase-int(phase) )
            phi.append( phase%1 )
        
        # Sort the pairs (phase,mag) by phase time in ascending order
        zipped                      = zip(lc.flux.value, phi)
        phase_folded_sorted_in_time = sorted(zipped, key=lambda x: x[1])
        phi_sort                    = []
        mag_sort                    = []
        for i in range(len(phase_folded_sorted_in_time)):
            phi_sort.append(phase_folded_sorted_in_time[i][1])
            mag_sort.append(phase_folded_sorted_in_time[i][0])
        
        # Compute the phase at the previous cycle (i.e. [-1,0]) in order to have a clear picture of the shape of the entire cycle [-1,1]
        phi1 = []
        for i in range(len(phi_sort)):
            phi1.append( phi_sort[i]-1 )
        
        # Return a LightCurve object consisting in the phase folded light curve over an entire cycle
        full_phase  = phi1 + phi_sort
        full_flux   = mag_sort + mag_sort

        if phase_range > 1:
            raise Exception("Range must be in (0,1]")

        out_phase   = []
        out_flux    = []
        for flux_i, phase_i in sorted( zip(full_flux, full_phase), key=lambda x: x[1] ):
            if phase_i > -phase_range and phase_i < phase_range:
                out_phase.append( phase_i )
                out_flux.append( flux_i )

        lc_phased = lk.LightCurve(time = out_phase, flux = out_flux)
        return lc_phased
    
    def lc_bin(self, lc_phased, time_bin_size_min=30, select_bin_size=False, bins=201):
        """
        Bin the phase folded data to make the transit more obvious.
        Input:
            - lc_phased: LightCurve object. The phase folded light curve.
            - time_bin_size_min: the width of the bins (in minutes). 
                                The higher the value, the smoother the binned signal is. 
                                TESS QLP data:  10 or 30 min, 
                                TESS SPOC data:  2 min, 
                                Kepler data:    30min
        Output:
            - A LightCurve object, consisting in the phase folded and binned signal. 
              The length of the output light curve depends on the target star.
        """
        if len(lc_phased.time.value) < bins:
            raise ValueError(f"Cannot bin into {bins} bins: lightcurve only has {len(lc_phased.time.value)} data points.")

        if select_bin_size:
            return lc_phased.bin(bins=bins)
        # convert the binning time from minutes to days
        observation_cadence_days = time_bin_size_min / 24 / 60
        return lc_phased.bin(observation_cadence_days)

    def __interpolate(self, inp, fi):
        """
            The following two methods interpolate the phase folded binned light curve in order to fix its length to a specific value.
            Source: https://stackoverflow.com/questions/44238581/interpolate-list-to-specific-length
            Input:
                - lc_phased_binned: A LightCurve object consisting in the phase folded binned light curve.
                - n_bins: the length of the output LightCurve object (e.g. the number of the global view bins)
        """
        i, f    = int(fi // 1), fi % 1   # Split floating-point index into whole & fractional parts.
        j       = i+1 if f > 0 else i       # Avoid index error.
        return  (1-f) * inp[i] + f * inp[j]

    def __set_global_view_length(self, lc_phased_binned, n_bins):
        inp_time, inp_flux  = lc_phased_binned.time.value, lc_phased_binned.flux.value
        delta               = (len(inp_flux)-1) / (n_bins-1)

        global_view_flux    = [self.__interpolate(inp_flux, i * delta) for i in range(n_bins)]
        global_view_time    = [self.__interpolate(inp_time, i * delta) for i in range(n_bins)]

        return lk.LightCurve(time = global_view_time, flux = global_view_flux)

    def preprocessing_pipeline(self, lc_collection, period, t0, duration):
        """
            The entire data preparation pipeline. 
            Input:
                - lc_collection (class lk.Lightcurve): the raw input light curve
                - period: transit period, in days
                - t0: transit epoch, corresponding to the time of the first detected transit
                - duration: transit duration, in hours
        """
        # Create the transit mask in order to preserve transits when detrending
        transit_mask            = lc_collection.create_transit_mask(period=period, transit_time=t0, duration=duration)
        
        # flattening, phase folding and binning the light curve
        lc_flatten              = self.lc_flattening(lc_collection, transit_mask)
        lc_phased               = self.lc_phase_folding(lc_flatten, t0, period, 0.8)
        lc_phased_binned        = self.lc_bin(lc_phased, self.sampling_rate)
        lc_global               = self.__set_global_view_length(lc_phased_binned, self.global_view_length)
        
        # Clean the global view from any NaN and normalize to 0 median and minimum transit depth to -1
        lc_global_flux_clean    = self._interpolate_nan(lc_global.flux.value)
        lc_global_flux_scaled   = self.__zero_median_fixed_depth(lc_global_flux_clean)

        return lk.LightCurve(time = lc_global.time.value, flux = lc_global_flux_scaled)
    
    def preprocessing_pipeline_from_binning(self, lc_phased):
        """
            Data preparation pipeline, from data binning. 
            Input:
                - lc_phased (class lk.Lightcurve): the phase folded light curve
                - fixed_depth: choose whether to standardize transit depth to -1
        """
        try:
            lc_phased_binned            = self.lc_bin(lc_phased, self.sampling_rate)
            lc_global                   = self.__set_global_view_length(lc_phased_binned, self.global_view_length)
            
            # Clean the global view from any NaN and normalize to 0 median and minimum transit depth to -1
            lc_global_flux_clean        = self._interpolate_nan(lc_global.flux.value)
            if self.fixed_depth:
                lc_global_flux_scaled   = self.__zero_median_fixed_depth(lc_global_flux_clean)
            else:
                lc_global_flux_scaled   = self._zero_median(lc_global_flux_clean)
        except Exception as e:
            print(f'[ERROR] In preprocessing_pipeline_from_binning(). The following exception was raised: {e}')

        return lk.LightCurve(time = lc_global.time.value, flux = lc_global_flux_scaled)

    def save_output_to_csv(self, output_csv_file, global_views, global_views_id, global_views_labels):
        """
            Save lightcurves to CSV.
        """
        with open(output_csv_file, mode='w', newline='') as file: # Aggiunto newline='' per evitare righe vuote
            writer = csv.writer(file)
            
            # Header 
            # Determina la lunghezza dei dati di flusso per creare l'intestazione dinamicamente
            if global_views:
                num_flux_points = len(global_views[0])
                header = [f'{FittedEventsColumns.PHASE_FLUX}{j}' for j in range(num_flux_points)] + [FittedEventsColumns.EVENT_ID] + [FittedEventsColumns.LABEL]
                writer.writerow(header)

            for i in range(len(global_views)):
                # La riga deve essere: [PHASE_FLUX_1, PHASE_FLUX_2, ..., EVENT_ID, PHASE_FLUX_N, LABEL]
                row = list(global_views[i]) + [global_views_id[i]] + [global_views_labels[i]]
                writer.writerow(row)

    def __del__(self):
        print('Destructor for DataProcessor')


if __name__ == '__main__':
    print('Class DataProcessor')
