<?php
/**
 * Trigger ML analysis for a couple by calling the local Flask-backed API.
 * Used by questionnaire flow once both partners have submitted.
 */

function trigger_ml_analysis($access_id) {
    if (!$access_id) {
        return false;
    }

    // Build API URL
    $api_url = __DIR__ . '/ml_api.php';

    // Prefer calling via HTTP to avoid include/session context issues
    // but since this runs server-side, we can invoke ml_api.php directly with cURL to localhost
    $endpoint = 'http://127.0.0.1' . dirname($_SERVER['SCRIPT_NAME'] ?? '/pmoc.bccbsis.com/ml_model') . '/ml_model/ml_api.php?action=analyze';

    // Fallback if SCRIPT_NAME is not reliable (e.g., CLI contexts)
    if (empty(parse_url($endpoint, PHP_URL_HOST))) {
        $endpoint = 'http://127.0.0.1/pmoc.bccbsis.com/ml_model/ml_api.php?action=analyze';
    }

    $payload = [ 'access_id' => $access_id ];

    try {
        $ch = curl_init($endpoint);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, 60);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);

        $response = curl_exec($ch);
        $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $curl_error = curl_error($ch);
        // curl_close() is deprecated in PHP 8.0+ - use unset() instead
        unset($ch);

        if ($response === false || !empty($curl_error)) {
            error_log('trigger_ml_analysis: cURL error: ' . $curl_error);
            return false;
        }

        if ($http_code !== 200) {
            error_log('trigger_ml_analysis: HTTP ' . $http_code . ' response: ' . substr((string)$response, 0, 200));
            return false;
        }

        $decoded = json_decode($response, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            error_log('trigger_ml_analysis: JSON decode error: ' . json_last_error_msg());
            return false;
        }

        // Consider success if API returns status success
        return isset($decoded['status']) && $decoded['status'] === 'success';
    } catch (\Throwable $e) {
        error_log('trigger_ml_analysis exception: ' . $e->getMessage());
        return false;
    }
}

?>


