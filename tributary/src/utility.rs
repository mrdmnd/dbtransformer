/// Timestamp encoding dimension: 5 cyclical pairs (sin/cos) + 1 linear = 11.
pub const TIMESTAMP_DIM: usize = 11;

/// Expand a raw epoch timestamp into the full 11-dimensional encoding.
/// Call this during sampling, not during preprocessing.
///
/// Returns: [sin_min, cos_min, sin_hr, cos_hr, sin_dow, cos_dow,
///           sin_doy, cos_doy, sin_month, cos_month, zscore_epoch]
#[inline]
pub fn expand_timestamp(epoch_secs: f32, mean: f64, std: f64) -> [f32; TIMESTAMP_DIM] {
    use std::f64::consts::TAU;

    let epoch_secs = epoch_secs as f64;

    const SECS_PER_MINUTE: f64 = 60.0;
    const SECS_PER_HOUR: f64 = 3600.0;
    const SECS_PER_DAY: f64 = 86400.0;
    const DAYS_PER_YEAR: f64 = 365.25;

    let days_since_epoch = epoch_secs / SECS_PER_DAY;
    let secs_today = epoch_secs.rem_euclid(SECS_PER_DAY);

    let minute_of_hour = secs_today.rem_euclid(SECS_PER_HOUR) / SECS_PER_MINUTE;
    let hour_of_day = secs_today / SECS_PER_HOUR;
    let day_of_week = (days_since_epoch + 4.0).rem_euclid(7.0); // Jan 1 1970 = Thursday
    let day_of_year = days_since_epoch.rem_euclid(DAYS_PER_YEAR);
    let month = day_of_year / (DAYS_PER_YEAR / 12.0);

    let minute_angle = TAU * minute_of_hour / 60.0;
    let hour_angle = TAU * hour_of_day / 24.0;
    let dow_angle = TAU * day_of_week / 7.0;
    let doy_angle = TAU * day_of_year / DAYS_PER_YEAR;
    let month_angle = TAU * month / 12.0;

    let std_safe = std.max(1e-8);
    let epoch_zscore = (epoch_secs - mean) / std_safe;

    [
        minute_angle.sin() as f32,
        minute_angle.cos() as f32,
        hour_angle.sin() as f32,
        hour_angle.cos() as f32,
        dow_angle.sin() as f32,
        dow_angle.cos() as f32,
        doy_angle.sin() as f32,
        doy_angle.cos() as f32,
        month_angle.sin() as f32,
        month_angle.cos() as f32,
        epoch_zscore as f32,
    ]
}
