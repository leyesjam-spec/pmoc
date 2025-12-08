<?php
/**
 * ML + AI Dashboard
 * Interface for managing and viewing AI counseling recommendations
 */

require_once '../includes/session.php';
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Counseling Topics Recommendations - BCPDO System</title>
    <!-- Font Awesome CDN for sorting icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<?php include '../includes/header.php'; ?>
</head>

<body class="hold-transition sidebar-mini layout-fixed">
    <div class="wrapper">
<?php include '../includes/navbar.php'; ?>
<?php include '../includes/sidebar.php'; ?>

<div class="content-wrapper">
  <section class="content-header">
    <div class="container-fluid">
      <div class="row mb-2">
        <div class="col-sm-6">
          <h1>Counseling Topics Recommendations</h1>
        </div>
        <div class="col-sm-6">
          <ol class="breadcrumb float-sm-right">
            <li class="breadcrumb-item"><a href="../index.php">Home</a></li>
            <li class="breadcrumb-item active">Counseling Topics</li>
          </ol>
        </div>
      </div>
    </div>
  </section>

  <section class="content">
    <div class="container-fluid">

      <!-- Action Buttons (Start Flask removed) -->
      <div class="row mb-4">
        <div class="col-12">
          <div class="btn-group" role="group">
            <button id="trainModelsBtn" class="btn btn-primary" title="Train the counseling topics prediction models (one-time setup)">
              <i class="fas fa-brain mr-1"></i>Train Models
            </button>
            <button id="analyzeAllBtn" class="btn btn-success" title="Generate counseling topics recommendations for all couples">
              <i class="fas fa-sync-alt mr-1"></i>Analyze All Couples
            </button>
          </div>
        </div>
      </div>

      <!-- Summary Cards -->
      <div class="row mb-4">
        <div class="col-lg-3 col-6">
          <div class="small-box bg-info">
            <div class="inner">
              <h3 id="totalCouples">0</h3>
              <p>Total Couples</p>
            </div>
            <div class="icon">
              <i class="fas fa-users"></i>
            </div>
          </div>
        </div>
        <div class="col-lg-3 col-6">
          <div class="small-box bg-success">
            <div class="inner">
              <h3 id="lowRiskCount">0</h3>
              <p>Low Risk</p>
            </div>
            <div class="icon">
              <i class="fas fa-check-circle"></i>
            </div>
          </div>
        </div>
        <div class="col-lg-3 col-6">
          <div class="small-box bg-warning">
            <div class="inner">
              <h3 id="mediumRiskCount">0</h3>
              <p>Medium Risk</p>
            </div>
            <div class="icon">
              <i class="fas fa-exclamation-triangle"></i>
            </div>
          </div>
        </div>
        <div class="col-lg-3 col-6">
          <div class="small-box bg-danger">
            <div class="inner">
              <h3 id="highRiskCount">0</h3>
              <p>High Risk</p>
            </div>
            <div class="icon">
              <i class="fas fa-exclamation-circle"></i>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Counseling Topics Recommendations Table -->
      <div class="row">
        <div class="col-12">
          <div class="card">
            <div class="card-header">
              <h3 class="card-title">
                <i class="fas fa-lightbulb mr-2"></i>Counseling Topics Recommendations
              </h3>
              <!-- Export buttons removed -->
            </div>
            <div class="card-body">
              <!-- Risk Filters -->
              <div class="mb-3" id="riskFilterBar">
                <div class="btn-group" role="group" aria-label="Risk filters">
                  <button type="button" class="btn btn-secondary btn-sm" id="riskFilterAll"><i class="fas fa-list mr-1"></i>All</button>
                  <button type="button" class="btn btn-success btn-sm" id="riskFilterLow"><i class="fas fa-check-circle mr-1"></i>Low</button>
                  <button type="button" class="btn btn-warning btn-sm" id="riskFilterMedium"><i class="fas fa-exclamation-triangle mr-1"></i>Medium</button>
                  <button type="button" class="btn btn-danger btn-sm" id="riskFilterHigh"><i class="fas fa-exclamation-circle mr-1"></i>High</button>
                </div>
              </div>
              
              <div id="loadingSpinner" class="text-center py-4">
                <i class="fas fa-spinner fa-spin fa-3x text-primary"></i>
                <p class="mt-3">Loading counseling topics recommendations...</p>
              </div>
              <div id="recommendationsTableWrapper" style="display: none;">
                <table id="recommendationsTable" class="table table-bordered table-hover table-striped">
                  <thead>
                    <tr>
                      <th>Couple Name</th>
                      <th>Risk Level</th>
                      <th class="d-none">RiskText</th>
                      <th>ML Confidence</th>
                      <th>Top Priority</th>
                      <th>Categories</th>
                      <th>Generated</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody id="recommendationsTableBody">
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</div>

<?php include '../includes/footer.php'; ?>
    </div>

<?php include '../includes/scripts.php'; ?>

<!-- SweetAlert2 for notifications -->
<script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>

<!-- SheetJS for Excel export -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>

<script>
$(document).ready(function() {
  let couplesState = [];
  let filteredCouples = [];
  let riskFilterLevel = null; // null means no filter (All)
  
  const loadingSpinner = document.getElementById('loadingSpinner');
  const recommendationsTableWrapper = document.getElementById('recommendationsTableWrapper');
  const recommendationsTableBody = document.getElementById('recommendationsTableBody');
  let dataTable = null;
  
  // Initialize dashboard
  init();
  
  async function init() {
    await loadCouples();
    await renderDashboard();
  }

  function badge(level) {
    const color = level === 'High' ? 'danger' : (level === 'Medium' ? 'warning' : 'success');
    return '<span class="badge badge-' + color + '">' + level + '</span>';
  }


  async function loadCouples() {
    try {
    const resp = await fetch('../couple_list/get_couples.php', { credentials: 'same-origin' });
    const data = await resp.json();
    if (!data.success) throw new Error('Failed to load couples');
    return data.data;
    } catch (error) {
      console.error('Error loading couples:', error);
      return [];
    }
  }

  async function getAnalysisResults(accessId) {
    const resp = await fetch('./ml_api.php?action=get_analysis&access_id=' + accessId, { credentials: 'same-origin' });
    const data = await resp.json();
    if (data.status !== 'success') throw new Error(data.message || 'Failed to fetch analysis');
    return data;
  }

  async function renderDashboard() {
    try {
      // Reset state
      couplesState.length = 0;
      
      // Load couples
      const couples = await loadCouples();
      couplesState = couples;
      
      // Process each couple
      const processedCouples = [];
      
      for (const couple of couples) {
        try {
          const analysis = await getAnalysisResults(couple.access_id);
          
          if (analysis.analyzed) {
            // Couple has been analyzed
            processedCouples.push({
              ...couple,
              analysis: analysis
            });
          } else {
            // Couple not yet analyzed
            processedCouples.push({
              ...couple,
              analysis: null
            });
          }
        } catch (error) {
          console.error(`Failed to fetch analysis for couple ${couple.access_id}:`, error);
          // Add couple without analysis
          processedCouples.push({
            ...couple,
            analysis: null
          });
        }
      }
      
      // Update filtered couples
      filteredCouples = processedCouples;
      
      // Update summary cards
      updateSummaryCards(processedCouples);
      
      // Render table
      renderTable(processedCouples);
      
      // Hide loading spinner and show table
      loadingSpinner.style.display = 'none';
      recommendationsTableWrapper.style.display = 'block';
      
    } catch (error) {
      console.error('Dashboard error:', error);
      loadingSpinner.innerHTML = '<p class="text-danger">Error loading counseling topics recommendations</p>';
    }
  }
  
  function updateSummaryCards(couples) {
    const total = couples.length;
    const lowRisk = couples.filter(c => c.analysis?.risk_level === 'Low').length;
    const mediumRisk = couples.filter(c => c.analysis?.risk_level === 'Medium').length;
    const highRisk = couples.filter(c => c.analysis?.risk_level === 'High').length;
    
    document.getElementById('totalCouples').textContent = total;
    document.getElementById('lowRiskCount').textContent = lowRisk;
    document.getElementById('mediumRiskCount').textContent = mediumRisk;
    document.getElementById('highRiskCount').textContent = highRisk;
  }
  
  function renderTable(couples) {
    // Destroy existing DataTable if it exists
    if (dataTable) {
      dataTable.destroy();
    }
    
    const tbody = recommendationsTableBody;
    tbody.innerHTML = '';
    
    couples.forEach(couple => {
      const analysis = couple.analysis;
      const riskLevel = analysis?.risk_level || 'Unknown';
      const mlConfidence = analysis?.ml_confidence || 0;
      const mlConfidencePercent = (mlConfidence * 100).toFixed(1) + '%';
      const focusCategories = analysis?.focus_categories || [];
      // Use updated_at if available (for re-analyzed couples), otherwise use generated_at
      const timestamp = analysis?.updated_at || analysis?.generated_at;
      let generatedAt = 'Not analyzed';
      if (timestamp) {
        // Parse the timestamp - MySQL timestamps are in server timezone
        // Create date object and convert to local timezone
        const date = new Date(timestamp);
        // Format with proper timezone handling
        generatedAt = date.toLocaleString('en-US', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          hour12: true,
          timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
        });
      }
      
      // Create ML confidence badge with color based on percentage
      let confidenceBadgeClass = 'secondary';
      if (mlConfidence > 0.7) confidenceBadgeClass = 'danger';
      else if (mlConfidence > 0.4) confidenceBadgeClass = 'warning';
      else if (mlConfidence > 0.2) confidenceBadgeClass = 'info';
      else if (mlConfidence > 0) confidenceBadgeClass = 'success';
      
      const confidenceBadge = `<span class="badge badge-${confidenceBadgeClass}">${mlConfidencePercent}</span>`;
      
      // Get highest priority category
      let topPriorityHTML = '<span class="text-muted">None</span>';
      if (focusCategories.length > 0) {
        const topCategory = focusCategories[0];
        const priorityColor = topCategory.priority === 'Critical' ? 'danger' : 
                            (topCategory.priority === 'High' ? 'warning' : 
                            (topCategory.priority === 'Moderate' ? 'warning' : 
                            (topCategory.priority === 'Low' ? 'success' : 'secondary')));
        const score = (topCategory.score * 100).toFixed(0);
        topPriorityHTML = `
          <span class="badge badge-${priorityColor} mb-1">${topCategory.priority}</span><br>
          <strong>${topCategory.name}</strong><br>
          <small class="text-muted">Score: ${score}%</small>
        `;
      }
      
      // Create expandable categories list
      let categoriesHTML = '';
      if (focusCategories.length > 0) {
        categoriesHTML = `<span class="badge badge-info">${focusCategories.length} categories</span>`;
        if (focusCategories.length > 1) {
          categoriesHTML += ` <a href="#" class="show-categories" data-access-id="${couple.access_id}">
            <i class="fas fa-chevron-down"></i>
          </a>`;
        }
      } else {
        categoriesHTML = '<span class="text-muted">None</span>';
      }
      
      const row = document.createElement('tr');
      row.setAttribute('data-access-id', couple.access_id);
      row.innerHTML = `
        <td><strong>${couple.couple_names}</strong></td>
        <td class="text-center">${badge(riskLevel)}</td>
        <td class="d-none">${riskLevel}</td>
        <td class="text-center">${confidenceBadge}</td>
        <td>${topPriorityHTML}</td>
        <td class="text-center">${categoriesHTML}</td>
        <td><small>${generatedAt}</small></td>
        <td class="text-center">
          <a href="./view_ml_recommendations.php?access_id=${couple.access_id}" 
             class="btn btn-outline-primary btn-sm" title="View Details">
            <i class="fas fa-eye mr-1"></i>View
          </a>
        </td>
      `;
      
      tbody.appendChild(row);
      
      // Store full categories data for expansion
      row.categoryData = focusCategories;
    });
    
    // Register a custom filter once to filter by RiskText using riskFilterLevel
    if (!window.__riskFilterRegistered) {
      $.fn.dataTable.ext.search.push(function(settings, data /*, dataIndex */) {
        // data is an array of column display text; RiskText is at index 2
        if (!riskFilterLevel || riskFilterLevel === 'All') {
          return true;
        }
        const riskText = (data[2] || '').trim();
        return riskText === riskFilterLevel;
      });
      window.__riskFilterRegistered = true;
    }

    // Initialize DataTable with options
    dataTable = $('#recommendationsTable').DataTable({
      "responsive": true,
      "lengthChange": true,
      "autoWidth": false,
      "pageLength": 10,
      "stateSave": false, // Disable state saving to prevent filter persistence issues
      "order": [[6, "desc"]], // Sort by Generated date descending (newest first)
      "language": {
        "search": "Search:",
        "lengthMenu": "Show _MENU_ entries",
        "info": "Showing _START_ to _END_ of _TOTAL_ entries",
        "infoEmpty": "No entries available",
        "infoFiltered": "(filtered from _MAX_ total entries)",
        "zeroRecords": "No matching records found"
      },
      "columnDefs": [
        { "visible": false, "targets": 2 }, // Hide RiskText column
        { "orderable": false, "targets": [5, 7] }, // Disable sorting on Categories and Actions
        { "width": "15%", "targets": 0 }, // Couple Name
        { "width": "10%", "targets": 1 }, // Risk Level
        { "width": "10%", "targets": 3 }, // ML Confidence
        { "width": "25%", "targets": 4 }, // Top Priority
        { "width": "10%", "targets": 5 }, // Categories
        { "width": "15%", "targets": 6 }, // Generated
        { "width": "10%", "targets": 7 }  // Actions
      ],
      "dom": '<"row"<"col-sm-12 col-md-6"l><"col-sm-12 col-md-6"f>>rtip',
      "initComplete": function() {
        // Fix sorting icons by using DataTables built-in icons
        setTimeout(function() {
          $('.dataTables_wrapper .dataTables_sorting').each(function() {
            var $this = $(this);
            var text = $this.text();
            
            // Remove any problematic text
            text = text.replace(/aa\+/g, '').replace(/[↑↓⇅⇈⇊⇧⇩⇨⇦⇨⇩⇧]/g, '').trim();
            $this.text(text);
            
            // Add proper sorting indicators based on class
            if ($this.hasClass('sorting_asc')) {
              $this.append(' <span style="font-size: 12px;">▲</span>');
            } else if ($this.hasClass('sorting_desc')) {
              $this.append(' <span style="font-size: 12px;">▼</span>');
            } else {
              $this.append(' <span style="font-size: 12px;">⇅</span>');
            }
          });
        }, 100);
      }
    });
    

    // Add category expansion functionality
    $('#recommendationsTable tbody').on('click', '.show-categories', function(e) {
      e.preventDefault();
      const row = $(this).closest('tr');
      const categoryData = row[0].categoryData;
      
      if (categoryData && categoryData.length > 0) {
        let categoriesDetail = '<div class="p-3"><h5>All Categories:</h5><ul class="list-unstyled">';
        categoryData.forEach(cat => {
          const priorityColor = cat.priority === 'Critical' ? 'danger' : 
                              (cat.priority === 'High' ? 'warning' : 
                              (cat.priority === 'Moderate' ? 'warning' : 
                              (cat.priority === 'Low' ? 'success' : 'secondary')));
          const score = (cat.score * 100).toFixed(0);
          categoriesDetail += `
            <li class="mb-2">
              <span class="badge badge-${priorityColor}">${cat.priority}</span>
              <strong>${cat.name}</strong>
              <small class="text-muted">(Score: ${score}%)</small>
            </li>
          `;
        });
        categoriesDetail += '</ul></div>';
        
        Swal.fire({
          title: 'All Priority Categories',
          html: categoriesDetail,
          width: '600px',
          confirmButtonText: 'Close'
        });
      }
    });
  }
  
  // Event listeners
  
  // Risk filter buttons
  function applyRiskFilter(level) {
    if (!dataTable) return;
    riskFilterLevel = level || null; // null or 'All' clears the filter
    // Clear global and column searches to avoid interfering with custom filter
    dataTable.search('', false, false);
    dataTable.columns().every(function() { this.search('', false, false); });
    const input = document.querySelector('#recommendationsTable_filter input');
    if (input) input.value = '';
    dataTable.order([]);
    dataTable.page('first');
    dataTable.draw();
  }
  $('#riskFilterAll').on('click', function(e){ e.preventDefault(); applyRiskFilter(null); });
  $('#riskFilterLow').on('click', function(e){ e.preventDefault(); applyRiskFilter('Low'); });
  $('#riskFilterMedium').on('click', function(e){ e.preventDefault(); applyRiskFilter('Medium'); });
  $('#riskFilterHigh').on('click', function(e){ e.preventDefault(); applyRiskFilter('High'); });

  // Train models button - now uses async polling with modal
  $('#trainModelsBtn').on('click', async function(){
    const btn = $(this);
    btn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin mr-1"></i>Starting...');
    
    try {
      // Show training modal
      if (typeof Swal !== 'undefined') {
        Swal.fire({
          title: 'Training ML Models',
          html: `
            <div style="text-align: center;">
              <p>Starting training...</p>
              <div style="width: 100%; background-color: #f0f0f0; border-radius: 10px; margin: 15px 0; overflow: hidden;">
                <div id="training-progress-bar" style="width: 0%; height: 25px; background: linear-gradient(90deg, #4CAF50, #45a049); transition: width 0.3s ease; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                  <span id="training-progress-text">0%</span>
                </div>
              </div>
              <small>This may take a few minutes</small>
            </div>
          `,
          allowOutsideClick: false,
          showConfirmButton: false,
          allowEscapeKey: false,
          didOpen: () => {
            // Don't show default loading spinner, we have custom progress bar
          }
        });
      }
      
      // Start training (returns immediately)
      const startResponse = await fetch('./ml_api.php?action=train', { method: 'POST' });
      const startData = await startResponse.json();
      
      if (startData.status !== 'success') {
        // Check if it's a 400 error (training already in progress)
        if (startResponse.status === 400) {
          throw new Error(startData.message || 'Training is already in progress. Please wait for the current training to complete.');
        }
        throw new Error(startData.message || 'Failed to start training');
      }
      
      // Poll for training status
      // REDUCED FREQUENCY: Changed from 1 second to 3 seconds to reduce database connection load
      let pollCount = 0;
      const maxPolls = 200; // ~10 minutes max (3 second intervals: 200 * 3 = 600 seconds)
      const pollInterval = 3000; // 3 seconds (reduced from 1 second to save database connections)
      
      const pollStatus = async () => {
        try {
          const statusResponse = await fetch('./ml_api.php?action=training_status');
          const statusData = await statusResponse.json();
          
          if (statusData.status === 'success') {
            const progress = statusData.progress || 0;
            const message = statusData.message || 'Training in progress...';
            
            // Update modal with progress
            if (typeof Swal !== 'undefined') {
              // Update progress bar
              const progressBar = document.getElementById('training-progress-bar');
              const progressText = document.getElementById('training-progress-text');
              if (progressBar && progressText) {
                progressBar.style.width = progress + '%';
                progressText.textContent = progress + '%';
              }
              
              // Update message
              Swal.update({
                html: `
                  <div style="text-align: center;">
                    <p>${message}</p>
                    <div style="width: 100%; background-color: #f0f0f0; border-radius: 10px; margin: 15px 0; overflow: hidden;">
                      <div id="training-progress-bar" style="width: ${progress}%; height: 25px; background: linear-gradient(90deg, #4CAF50, #45a049); transition: width 0.3s ease; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                        <span id="training-progress-text">${progress}%</span>
                      </div>
                    </div>
                    <small>This may take a few minutes</small>
                  </div>
                `,
                showConfirmButton: false
              });
            }
            
            if (!statusData.in_progress) {
              // Training completed
              if (statusData.error) {
                if (typeof Swal !== 'undefined') {
                  Swal.fire({
                    icon: 'error',
                    title: 'Training Failed',
                    text: statusData.error,
                    showConfirmButton: true
                  });
                }
                btn.prop('disabled', false).html('<i class="fas fa-brain mr-1"></i>Train Models');
              } else {
                if (typeof Swal !== 'undefined') {
                  Swal.fire({
                    icon: 'success',
                    title: 'Training Complete!',
                    html: `<p>${statusData.message || 'ML models trained successfully'}</p><br><small>Models are ready to use</small>`,
                    showConfirmButton: true,
                    confirmButtonText: 'OK'
                  });
                }
                btn.prop('disabled', false).html('<i class="fas fa-brain mr-1"></i>Train Models');
              }
              return; // Stop polling
            }
            
            // Continue polling with adaptive interval (slower when progress is slow)
            pollCount++;
            if (pollCount < maxPolls) {
              // Adaptive polling: if progress hasn't changed much, poll less frequently
              // This reduces database connection load
              const adaptiveInterval = pollInterval;
              setTimeout(pollStatus, adaptiveInterval);
            } else {
              throw new Error('Training timeout - please check server logs');
            }
          } else {
            throw new Error(statusData.message || 'Failed to get training status');
          }
        } catch (error) {
          if (error.message.includes('timeout')) {
            if (typeof Swal !== 'undefined') {
              Swal.fire({
                icon: 'error',
                title: 'Training Timeout',
                text: error.message,
                showConfirmButton: true
              });
            }
            btn.prop('disabled', false).html('<i class="fas fa-brain mr-1"></i>Train Models');
            return;
          }
          // Retry on network error
          pollCount++;
          if (pollCount < maxPolls) {
            setTimeout(pollStatus, pollInterval);
          } else {
            if (typeof Swal !== 'undefined') {
              Swal.fire({
                icon: 'error',
                title: 'Training Status Check Failed',
                text: 'Training status check failed - please check server logs',
                showConfirmButton: true
              });
            }
            btn.prop('disabled', false).html('<i class="fas fa-brain mr-1"></i>Train Models');
          }
        }
      };
      
      // Start polling after a short delay
      setTimeout(pollStatus, pollInterval);
      
    } catch (error) {
      if (typeof Swal !== 'undefined') {
        Swal.fire({
          icon: 'error',
          title: 'Training Failed',
          html: `<p>${error.message}</p><br><small>Make sure the Flask service is running.</small>`,
          showConfirmButton: true
        });
      }
      btn.prop('disabled', false).html('<i class="fas fa-brain mr-1"></i>Train Models');
    }
  });

  // Analyze all couples button
  $('#analyzeAllBtn').on('click', async function(){
    const btn = $(this);
    btn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin mr-1"></i>Analyzing...');
    
    try {
      // Get all couple access_ids
      const access_ids = couplesState.map(c => c.access_id);
      
      if (access_ids.length === 0) {
        throw new Error('No couples found to analyze');
      }
      
      // Show progress message
      if (typeof Swal !== 'undefined') {
        Swal.fire({
          title: 'Analyzing Couples',
          html: `Analyzing ${access_ids.length} couples...<br><small>This may take a few minutes</small>`,
          allowOutsideClick: false,
          didOpen: () => {
            Swal.showLoading();
          }
        });
      }
      
      // Call batch analysis
      const formData = new FormData();
      formData.append('access_ids', JSON.stringify(access_ids));
      
      const response = await fetch('./ml_api.php?action=analyze_batch', {
        method: 'POST',
        body: formData,
        credentials: 'same-origin'
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        const results = data.results;
        
        if (typeof Swal !== 'undefined') {
          Swal.fire({
            icon: results.failed > 0 ? 'warning' : 'success',
            title: 'Analysis Complete!',
            html: `
              <p><strong>Analyzed:</strong> ${results.success} of ${results.total} couples</p>
              ${results.failed > 0 ? `<p class="text-warning"><strong>Failed:</strong> ${results.failed}</p>` : ''}
              ${results.errors.length > 0 ? `<details class="text-left mt-2"><summary>Error Details</summary><small>${results.errors.join('<br>')}</small></details>` : ''}
            `,
            showConfirmButton: true,
            confirmButtonText: 'Refresh Dashboard'
          }).then(() => {
            // Refresh dashboard to show new results
            location.reload();
          });
        } else {
          // Fallback: just reload
          location.reload();
        }
      } else {
        throw new Error(data.message || 'Batch analysis failed');
      }
    } catch (error) {
      if (typeof Swal !== 'undefined') {
        Swal.fire({
          icon: 'error',
          title: 'Analysis Failed',
          html: `<p>${error.message}</p><br><small>Make sure the Flask service is running.</small>`,
          showConfirmButton: true
        });
      }
      console.error('Batch analysis error:', error);
    } finally {
      btn.prop('disabled', false).html('<i class="fas fa-sync-alt mr-1"></i>Analyze All Couples');
    }
  });

  // Start Flask Service button removed
});
</script>

<style>
/* Summary Cards */
.small-box .icon {
  top: 10px;
  right: 10px;
}

/* Table Styling */
#recommendationsTable {
  font-size: 0.9rem;
}

#recommendationsTable thead th {
  background-color: #f8f9fa;
  border-top: none;
  font-weight: 600;
  vertical-align: middle;
}

#recommendationsTable tbody tr {
  transition: all 0.2s ease-in-out;
}

#recommendationsTable tbody tr:hover {
  background-color: #f1f3f5 !important;
  transform: scale(1.01);
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

/* Badges */
.badge {
  font-size: 0.85em;
  padding: 0.35em 0.6em;
  font-weight: 600;
}

/* Action Buttons */
.btn-group .btn {
  margin-right: 5px;
}

.show-categories {
  cursor: pointer;
  color: #007bff;
  text-decoration: none;
}

.show-categories:hover {
  color: #0056b3;
}

/* Loading Spinner */
#loadingSpinner {
  min-height: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

/* Footer */
.main-footer {
  margin-top: 20px;
  padding: 15px;
  background-color: #f4f6f9;
  border-top: 1px solid #dee2e6;
}

.content-wrapper {
  min-height: calc(100vh - 200px);
}

/* DataTables Custom Styling */
.dataTables_wrapper .dataTables_length select {
  padding: 0.25rem 0.5rem;
  border-radius: 0.2rem;
  min-width: 60px;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  background-color: #fff;
  border: 1px solid #ced4da;
}

.dataTables_wrapper .dataTables_filter input {
  border-radius: 0.2rem;
  padding: 0.25rem 0.5rem;
  margin-left: 0.5rem;
}


/* Print Styles */
@media print {
  .btn, .card-tools, .main-sidebar, .main-header, .content-header, 
  .dataTables_filter, .dataTables_length, .dataTables_info, .dataTables_paginate {
    display: none !important;
  }
  
  .content-wrapper {
    margin-left: 0 !important;
    padding: 0 !important;
  }
  
  .card {
    border: none !important;
    box-shadow: none !important;
  }
  
  #recommendationsTable {
    font-size: 10pt;
  }
  
  .badge {
    border: 1px solid #000;
    color: #000 !important;
    background-color: #fff !important;
  }
}

/* Fix sorting icons with Unicode arrows */
.dataTables_wrapper .dataTables_sorting {
  position: relative;
  cursor: pointer;
}

/* Hide default DataTables sorting indicators */
.dataTables_wrapper .dataTables_sorting:before,
.dataTables_wrapper .dataTables_sorting:after {
  display: none !important;
  content: none !important;
}

/* Hide any background sorting indicators */
.dataTables_wrapper .dataTables_sorting {
  background-image: none !important;
}

/* Style the Unicode sorting arrows */
.dataTables_wrapper .dataTables_sorting span {
  margin-left: 5px;
  color: #666;
  font-weight: normal;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  #recommendationsTable {
    font-size: 0.8rem;
  }
  
  .badge {
    font-size: 0.75em;
  }
}
</style>
</body>
</html>
